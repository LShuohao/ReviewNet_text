function makeLstmUnit(nIn, nHidden, dropout, maxT)
    --[[ Create attention LSTM unit, adapted from crnn/src/LSTMLayer.lua
    ARGS:
      - `nIn`      : integer, number of input dimensions
      - `nHidden`  : integer, number of hidden nodes
      - `dropout`  : boolean, if true apply dropout
    RETURNS:
      - `lstmUnit` : constructed LSTM unit (nngraph module)
    ]]

    dropout = dropout or 0
    -- there will 3 inputs: x (input), prev_c, prev_h--------------------------
    local x, prev_c, prev_h = nn.Identity()(), nn.Identity()(), nn.Identity()()
    local inputs = {x, prev_c, prev_h}
    -- apply dropout, if any
    if dropout > 0 then x = nn.Dropout(dropout)(x) end
    -- Construct the unit structure
    -- attention processing----------------------------------------------------
    local U_T  = nn.Reshape(-1, nIn, false)(x)               ---- Reshape the X to 26*64x512
    local U_F  = nn.Linear(nIn, nHidden)(U_T)                ---- line relation x
    local W_T  = nn.Linear(nHidden, nHidden)(prev_h)         ---- prev_h line relation
    local W_F  = nn.Replicate(maxT)(W_T)                     ---- replicate the W_T  
    local W    = nn.Reshape(-1, nHidden, false)(W_F)         ---- Reshape W to 26*64x256
    local tf   = nn.Tanh()(nn.CAddTable()({U_F, W}))         ---- tanh(W+U)
    local e_T  = nn.Linear(nHidden, 1)(tf)                   ---- line relation T
    local e_F  = nn.Reshape(maxT, -1, false)(e_T)            ---- Reshape back  
    local e    = nn.Transpose({1, 2})(e_F)                   ---- 26*64 to 64*26
    e = nn.SoftMax()(e)                                      ---- softmax and get the att para
    local E    = nn.Transpose({1, 2})(e)                     ---- 64*26 to 26*64
    local pa   = nn.Transpose({1, 3})(nn.Replicate(nIn)(E))  ---- 512*26*64 to 64*26*512
    local ctx  = nn.Transpose({1, 2})(x)                     ---- x(26*64*512) to 64*26*512
    local pctx = nn.Sum(2)(nn.CMulTable()({pa, ctx}))        ---- mul and to 64*512
    ---------------------------------------------------------------------------
    --------evaluate the input sums at once for attention and effictive--------    
    local i2h            = nn.Linear(nIn,     4*nHidden)(pctx)
    local h2h            = nn.Linear(nHidden, 4*nHidden)(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})
    -- decode the gates
    local sigmoid_chunk  = nn.Narrow(2, 1, 3*nHidden)(all_input_sums)
    sigmoid_chunk        = nn.Sigmoid()(sigmoid_chunk)
    local in_gate        = nn.Narrow(2,           1, nHidden)(sigmoid_chunk)
    local forget_gate    = nn.Narrow(2,   nHidden+1, nHidden)(sigmoid_chunk)
    local out_gate       = nn.Narrow(2, 2*nHidden+1, nHidden)(sigmoid_chunk)
    -- decode the write inputs
    local in_transform   = nn.Narrow(2, 3*nHidden+1, nHidden)(all_input_sums)
    in_transform         = nn.Tanh()(in_transform)
    -- perform the LSTM update
    local next_c         = nn.CAddTable()({
                               nn.CMulTable()({forget_gate, prev_c}),
                               nn.CMulTable()({in_gate    , in_transform})
                               })
    -- gated cells from the output
    local next_h         = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    -- y (output)
    local y              = nn.Identity()(next_h)
    ---------------------------------------------------------------------------
    -- there will be 3 outputs
    local outputs = {next_c, next_h, y}

    local lstmUnit = nn.gModule(inputs, outputs)
    return lstmUnit
end


local LstmLayer, parent = torch.class('nn.LstmLayer', 'nn.Module')


function LstmLayer:__init(nIn, nHidden, maxT, dropout, reverse)
    --[[
    ARGS:
      - `nIn`     : integer, number of input dimensions
      - `nHidden` : integer, number of hidden nodes
      - `maxT`    : integer, maximum length of input sequence
      - `dropout` : boolean, if true apply dropout
      - `reverse` : boolean, if true the sequence is traversed from the end to the start
    ]]
    parent.__init(self)

    self.dropout   = dropout or 0
    self.reverse   = reverse or false
    self.nHidden   = nHidden
    self.maxT      = maxT

    self.output    = {}
    self.gradInput = {}

    -- LSTM unit and clones
    self.lstmUnit  = makeLstmUnit(nIn, nHidden, self.dropout,maxT)
    self.clones    = {}

    -- LSTM states
    self.initState = {torch.CudaTensor(), torch.CudaTensor()} -- c, h

    self:reset()
end


function LstmLayer:reset(stdv)
    local params, _ = self:parameters()
    for i = 1, #params do
        if i % 2 == 1 then -- weight
            params[i]:uniform(-0.08, 0.08)
        else -- bias
            params[i]:zero()
        end
    end
end


function LstmLayer:type(type)
    assert(#self.clones == 0, 'Function type() should not be called after cloning.')
    self.lstmUnit:type(type)
    return self
end


function LstmLayer:parameters()
    return self.lstmUnit:parameters()
end


function LstmLayer:training()
    self.train = true
    self.lstmUnit:training()
    for t = 1, #self.clones do self.clones[t]:training() end
end


function LstmLayer:evaluate()
    self.train = false
    self.lstmUnit:evaluate()
    for t = 1, #self.clones do self.clones[t]:evaluate() end
end


function LstmLayer:updateOutput(input)
    if input:size(3) == 512 then
        T = 8
    else
        T = 26
    end
--    T = input:size(1)
--    print(T)
    local batchSize = input:size(2)
    self.initState[1]:resize(batchSize, self.nHidden):fill(0)
    self.initState[2]:resize(batchSize, self.nHidden):fill(0)
    if #self.clones == 0 then
        self.clones = cloneManyTimes(self.lstmUnit, T)
    end

    if not self.reverse then
        self.rnnState = {[0] = cloneList(self.initState, true)}
        for t = 1, T do
            local lst
            if self.train then
                -- print(t, input:size(1), input:size(2), input:size(3))
                lst = self.clones[t]:forward({input, unpack(self.rnnState[t-1])})
            else
                -- print(input:size(1),input:size(2), input:size(3))
                lst = self.lstmUnit:forward({input, unpack(self.rnnState[t-1])})
                lst = cloneList(lst)
            end
            self.rnnState[t] = {lst[1], lst[2]} -- next_c, next_h
            self.output[t] = lst[3]
        end
    else
        self.rnnState = {[T+1] = cloneList(self.initState, true)}
        for t = T, 1, -1 do
            local lst
            if self.train then
                lst = self.clones[t]:forward({input, unpack(self.rnnState[t+1])})
            else
                lst = self.lstmUnit:forward({input, unpack(self.rnnState[t+1])})
                lst = cloneList(lst)
            end
            self.rnnState[t] = {lst[1], lst[2]}
            self.output[t] = lst[3]
        end
    end
    return self.output
end


function LstmLayer:updateGradInput(input, gradOutput)
--    print(input:size(1))
--    print(#gradOutput)
--    assert(input:size(1) == #gradOutput)
    if input:size(3) == 512 then
        T = 8
    else
        T = 26
    end
    if not self.reverse then
        self.drnnState = {[T] = cloneList(self.initState, true)} -- zero gradient for the last frame
        for t = T, 1, -1 do
            local doutput_t = gradOutput[t]
            table.insert(self.drnnState[t], doutput_t) -- dnext_c, dnext_h, doutput_t
            local dlst = self.clones[t]:updateGradInput({input, unpack(self.rnnState[t-1])}, self.drnnState[t]) -- dx, dprev_c, dprev_h
            self.drnnState[t-1] = {dlst[2], dlst[3]}
            local gradInput = dlst[1]
            if t == T then
                self.gradInput = gradInput
            else
                self.gradInput:add(gradInput)
            end
        end
    else
        self.drnnState = {[1] = cloneList(self.initState, true)}
        for t = 1, T do
            local doutput_t = gradOutput[t]
            table.insert(self.drnnState[t], doutput_t)
            local dlst = self.clones[t]:updateGradInput({input, unpack(self.rnnState[t+1])}, self.drnnState[t])
            self.drnnState[t+1] = {dlst[2], dlst[3]}
            local gradInput = dlst[1]
            if t == 1 then
                self.gradInput = gradInput
            else
                self.gradInput:add(gradInput)
            end
        end
    end
    --print(self.gradInput:size())
    return self.gradInput
end


function LstmLayer:accGradParameters(input, gradOutput, scale)
    if input:size(3) == 512 then
        T = 8
    else
        T = 26
    end
    if not self.reverse then
        for t = 1, T do
            self.clones[t]:accGradParameters({input, unpack(self.rnnState[t-1])}, self.drnnState[t], scale)
        end
    else
        for t = T, 1, -1 do
            self.clones[t]:accGradParameters({input, unpack(self.rnnState[t+1])}, self.drnnState[t], scale)
        end
    end
end

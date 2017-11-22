hyperparameters
hidden nodes in nn - 1000

def forward(self, input_vector):
        out = self.input_linear(input_vector)
        out = F.tanh(out)
        out = self.output_linear(out)
        out = F.softmax(out)
        return out

word embedding - randomly initalized 300 dimension
tag embedding - 1-hot 127+1 vector
loss function - MSE loss
optimizer = optim.Adam(model.parameters(), lr=0.0005)
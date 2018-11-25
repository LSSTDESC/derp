from torch import nn

class Regressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Regressor, self).__init__()

        self.relu = nn.ReLU()
        self.bn_1 = nn.BatchNorm1d(12)
        self.bn_2 = nn.BatchNorm1d(100)
        self.bn_3 = nn.BatchNorm1d(200)
        self.bn_4 = nn.BatchNorm1d(400)
        
        self.fc_1 = nn.Linear(12, 100)
        self.fc_2 = nn.Linear(100, 200)
        self.fc_3 = nn.Linear(200, 400)
        self.fc_4 = nn.Linear(400, 9)
        
    def forward(self, x):
        y_pred = self.bn_1(x)
        y_pred = self.fc_1(y_pred)
        y_pred = self.relu(y_pred)
        
        y_pred = self.bn_2(y_pred)
        y_pred = self.fc_2(y_pred)
        y_pred = self.relu(y_pred)
        
        y_pred = self.bn_3(y_pred)
        y_pred = self.fc_3(y_pred)
        y_pred = self.relu(y_pred)
        
        y_pred = self.bn_4(y_pred)
        y_pred = self.fc_4(y_pred)
        return y_pred
    
if __name__=='__main__':
    # Test constructor
    regressor = Regressor(3, 5)
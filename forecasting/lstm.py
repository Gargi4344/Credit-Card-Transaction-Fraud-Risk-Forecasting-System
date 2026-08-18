from pathlib import Path
import numpy as np,torch
from torch import nn
class LSTMForecaster(nn.Module):
 def __init__(self,hidden=64,layers=2):
  super().__init__();self.rnn=nn.LSTM(1,hidden,layers,batch_first=True,dropout=.1);self.head=nn.Sequential(nn.Linear(hidden,32),nn.ReLU(),nn.Linear(32,1),nn.Sigmoid())
 def forward(self,x): return self.head(self.rnn(x)[0][:,-1,:])
def windows(v,lookback):
 X=[];y=[]
 for i in range(lookback,len(v)):X.append(v[i-lookback:i]);y.append(v[i])
 return torch.tensor(np.array(X),dtype=torch.float32).unsqueeze(-1),torch.tensor(np.array(y),dtype=torch.float32).unsqueeze(-1)
def train(v,epochs=40,lookback=21):
 torch.manual_seed(42);X,y=windows(np.asarray(v,dtype=np.float32),lookback);m=LSTMForecaster();opt=torch.optim.Adam(m.parameters(),lr=1e-3);lossfn=nn.MSELoss()
 for _ in range(epochs):
  opt.zero_grad();loss=lossfn(m(X),y);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 return m,float(loss.detach())
@torch.no_grad()
def forecast(m,history,horizon,lookback=21):
 m.eval();h=list(np.asarray(history,dtype=np.float32))
 for _ in range(horizon):
  x=torch.tensor(h[-lookback:],dtype=torch.float32).view(1,lookback,1);h.append(float(torch.clamp(m(x),0,1).item()))
 return np.array(h[-horizon:])
def save(m,path):Path(path).parent.mkdir(parents=True,exist_ok=True);torch.save(m.state_dict(),path)

from torch.utils.data import Dataset
        
class KinematicsDataset(Dataset):
    def __init__(self, robot, len_batch= 4096):
        self.len_batch = len_batch
        self.robot = robot
    
    def __getitem__(self, index):
        qx = self.robot.get_pair()
        return qx

    def __len__(self):
        return self.len_batch
    

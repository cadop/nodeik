import torch
import torch.nn as nn

import pytorch_lightning as pl

from pyquaternion import Quaternion
import numpy as np

from nodeik.utils import standard_normal_logprob
from nodeik.training.model_wrapper import ModelWrapper


class Config:
    """Convenience class that overwrites the default print function"""

    ROUND_AMT = 7

    def __init__(self, class_name: str):
        self.class_name = class_name

    def __str__(self) -> str:
        ret_str = f"{self.class_name}()\n"
        longest_key_len = len(max([key for key in self.__dict__], key=len))
        for key in self.__dict__:
            n_spaces = longest_key_len - len(key) + 1
            val = self.__dict__[key]
            if isinstance(val, float):
                ret_str += f"  {key}:{n_spaces*' '}{round(val, Config.ROUND_AMT)}\n"
            elif isinstance(val, int):
                ret_str += f"  {key}:{n_spaces*' '}{val}\n"
            elif isinstance(val, str):
                ret_str += f"  {key}:{n_spaces*' '}'{val}'\n"
            else:
                ret_str += f"  {key}:{n_spaces*' '} {type(val)} {val}\n"
        return ret_str


class ModelAccuracyResults(Config):
    def __init__(
        self,
        l2_errors,
        angular_errors
    ):
        # import pdb; pdb.set_trace()
        assert len(l2_errors) == len(angular_errors)
        samples_per_endpoint = l2_errors[0].shape[0]
        for l2_errors_i, ang_errs_i in zip(l2_errors, angular_errors):
            assert l2_errors_i.shape == ang_errs_i.shape
            assert l2_errors_i.shape[0] == samples_per_endpoint
        testset_size = len(l2_errors)

        union_l2_errs = np.array(l2_errors)
        union_angular_errs = np.array(angular_errors)
        assert union_l2_errs.size == testset_size * samples_per_endpoint
        assert union_angular_errs.size == testset_size * samples_per_endpoint

        max_l2_errors = [np.max(errs_i) for errs_i in l2_errors]
        max_ang_errors = [np.max(errs_i) for errs_i in angular_errors]
        assert len(max_l2_errors) == testset_size
        assert len(max_ang_errors) == testset_size

        # L2 Error
        self.ave_l2err = np.mean(union_l2_errs)
        self.median_l2err = np.median(union_l2_errs)
        self.std_l2errs = np.std(union_l2_errs)
        self.max_l2err = np.max(union_l2_errs)

        self.q1_l2err = np.quantile(union_l2_errs,0.25)
        self.q2_l2err = np.quantile(union_l2_errs,0.5)
        self.q3_l2err = np.quantile(union_l2_errs,0.75)
        # The average max l2 error on a returned batch of solutions
        self.ave_max_l2err = np.mean(max_l2_errors)
        # The standard deviation of the max l2 error for a returned batch of solutions
        self.std_max_l2err = np.std(max_l2_errors)

        # Angular error
        self.ave_angular_err = np.mean(union_angular_errs)
        self.median_angular_err = np.median(union_angular_errs)
        self.std_angular_err = np.std(union_angular_errs)
        self.max_angular_err = np.max(union_angular_errs)
        self.q1_angular_err = np.quantile(union_angular_errs,0.25)
        self.q2_angular_err = np.quantile(union_angular_errs,0.5)
        self.q3_angular_err = np.quantile(union_angular_errs,0.75)
        # The average max angular error on a returned batch of solutions
        self.ave_max_angular_err = np.mean(max_ang_errors)
        # The stnadard deviation of the max angular error for a returned batch of solutions
        self.std_max_angular_errs = np.std(max_ang_errors)

        # Evaluation diagnostics
        self.test_set_size = testset_size
        self.samples_per_endpoint = samples_per_endpoint

    def get_dict(self):
        return self.__dict__

    def __str__(self) -> str:
        nl = "\n"
        rnd = 4
        ret_str = nl + "ModelAccuracyResults()" + nl
        ret_str += f" ave_l2err:     {round(self.ave_l2err, rnd)} \t (std: {round(self.std_l2errs, rnd)})" + nl
        ret_str += f" max_l2err:   {round(self.max_l2err, rnd)}" + nl
        ret_str += f" ave_max_l2err: {round(self.ave_max_l2err, rnd)} \t (std: {round(self.std_max_l2err, rnd)})" + nl
        ret_str += " ----------" + nl
        ret_str += (
            f" ave_angular_err: {round(self.ave_angular_err, rnd)} \t (std: {round(self.std_angular_err, rnd)})" + nl
        )
        ret_str += f" max_angular_err:   {round(self.max_angular_err, rnd)}" + nl
        ret_str += (
            f" ave_max_angular_err: {round(self.ave_max_angular_err, rnd)} \t (std: {round(self.std_max_angular_errs, rnd)})"
            + nl
        )
        ret_str += (
            f" mean model runtime: {round(self.ave_model_runtime, 3)} \t ({self.test_set_size} target poses in testset, {self.samples_per_endpoint} samples generated per pose)"
            + nl
        )
        return ret_str

class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module, robot, std=1.0, state_dim=7, condition_dim = 7, num_samples=250):
        super().__init__()
        self.model = model
        self.iters = 0
        self.robot = robot
        self.model_wrapper = ModelWrapper(self.model, robot, self.device, std=std, dim_c=condition_dim, dim_x=state_dim)
        self.state_dim = state_dim
        self.condition_dim = condition_dim

        self.num_samples = num_samples
        
    def forward(self, xc):
        zero = torch.zeros(x.shape[0], 1).to(xc)
        x = xc[:self.state_dim]
        c = xc[-self.condition_dim:]
        return self.model(x,c,zero)
    
    def training_step(self, batch, batch_idx):
        self.iters += 1
        xc = batch
        
        zero = torch.zeros(xc.shape[0], 1).to(xc)

        x = xc[:, :self.state_dim]
        c = xc[:, -self.condition_dim:]
        z, delta_logp = self.model(x, c, zero)

        logpz = standard_normal_logprob(z).sum(1, keepdim=True)

        logpx = logpz - delta_logp
        loss = -torch.mean(logpx)


        # z2 = torch.normal(mean=0.0, std=1.0, size=(c.shape[0],7)).to(x)
        # # print('x',x )
        # # print(x.shape, c.shape, zero.shape)
        # ik_q, delta_logp = self.model(z2, c, zero, reverse=True)
        # ik_q = ik_q.cpu().detach().numpy()


        return {'loss': loss} 

    def validation_step(self, batch, batch_idx):
        # print('validation_step')
        self.iters += 1
        xc = batch
        
        zero = torch.zeros(xc.shape[0], 1).to(xc)

        joint = xc[:, :self.state_dim]
        ee_pose_target = xc[:, -self.condition_dim:]

        z, delta_logp = self.model(joint, ee_pose_target, zero)

        logpz = standard_normal_logprob(z).sum(1, keepdim=True)

        logpx = logpz - delta_logp
        loss = -torch.mean(logpx)

        ee_pose_target_np = ee_pose_target.cpu().detach().numpy()

        # with torch.inference_mode():
        p_errs = []
        q_errs = []
        for ee_pose_target_single in ee_pose_target_np:
            q_out, _ = self.model_wrapper.inverse_kinematics(ee_pose_target_single, self.num_samples) # 250
            x_out = self.model_wrapper.forward_kinematics(q_out)

            p_err = []
            q_err = []
            if x_out.shape[1] == 3: # 3 dof
                pass
            if x_out.shape[1] == 7: # single
            for a in x_out:
                b = ee_pose_target_single
                pos_norm = np.linalg.norm(a[:3] - b[:3])
                q1 = Quaternion(array=a[3:])
                q2 = Quaternion(array=b[3:])
                quat_norm = Quaternion.distance(q1,q2)
                p_err.append(pos_norm)
                q_err.append(quat_norm)
            # for d in diff:
            #     pos_norm = np.linalg.norm(d[:3])
            #     quat_norm = np.linalg.norm(d[3:])
                # print(pos_norm,quat_norm)
            elif x_out.shape[1] == 14: # dual
                for aa in x_out:
                    for i in range(2):    
                        a = aa[i*7:(i+1)*7]
                        b = ee_pose_target_single[i*7:(i+1)*7]
                        pos_norm = np.linalg.norm(a[:3] - b[:3])
                        q1 = Quaternion(array=a[3:])
                        q2 = Quaternion(array=b[3:])
                        quat_norm = Quaternion.distance(q1,q2)
                        p_err.append(pos_norm)
                        q_err.append(quat_norm)
            else:
                print('x_out.shape', x_out.shape)
                assert(False)
            p_errs.append(p_err)
            q_errs.append(q_err)

        result = ModelAccuracyResults(np.array(p_errs), np.array(q_errs))
        dicts = result.get_dict()
        results_dict = {"loss": loss}
        results_dict.update(dicts)
        
        self.log_dict(results_dict)
        # print('mean position    error:', np.array(p_errs).mean())
        # print('mean orientation error:', np.array(q_errs).mean())


        
        # z2 = torch.normal(mean=0.0, std=1.0, size=(c.shape[0],7)).to(x)
        # # print('x',x )
        # # print(x.shape, c.shape, zero.shape)
        # ik_q, delta_logp = self.model(z2, c, zero, reverse=True)
        # ik_q = ik_q.cpu().detach().numpy()

        

        metrics = {}
        
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=2e-3, weight_decay=1e-5)

    # def train_dataloader(self):
    #     return self.dataloader

    # def val_dataloader(self):
    #     return self.val_dataloader

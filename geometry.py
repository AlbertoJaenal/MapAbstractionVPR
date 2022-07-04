# Inspired on https://github.com/mingu6/ProbFiltersVPR/blob/master/src/geometry.py
import numpy as np
import math
from scipy.spatial.transform import Rotation

####################################################################################################
######################### ROTATION 2D
####################################################################################################

class Rotation2(Rotation):
    def apply2(self, t):
        try:
            if not self.single: length = len(self)
            else: length = 1
        except:
            if not self._single: length = len(self)
            else: length = 1
    
        if length == 1: 
            matrix = self.as_matrix()
            if len(matrix.shape) == 3: matrix = matrix[0]
            if len(t) == 1 or t.shape == (2,): # Case 1, 1
                return np.matmul(matrix[:2, :2], t)
            else: # Case 1, mul
                return np.array([np.matmul(matrix[:2, :2], tr) for tr in t])
        else:
            if len(t) == 1 or t.shape == (2,): # Case mul, 1
                return np.array([np.matmul(rot_mat[:2, :2], t.squeeze()) for rot_mat in self.as_matrix()])
            elif len(t) == length: # Case mul, mul
                return np.array([np.matmul(rot_mat[:2, :2], tr) for rot_mat, tr in zip(self.as_matrix(), t)])
            else:
                raise ValueError("Expected len(t) == {}, got {}".format(length, len(t)))
    
    def as_rotvec(self):
        try:
            if not self.single: length = len(self)
            else: length = 1
        except:
            if not self._single: length = len(self)
            else: length = 1
            
        output = super().as_rotvec()
        if length == 1:
            if len(output.shape) == 2:
                return output.squeeze()[-1].reshape([1, 1])
            else:
                return output.squeeze()[-1].reshape([1])
        else:
            return output[:, -1].reshape([-1, 1])

####################################################################################################
######################### POSE 2D 
####################################################################################################


class SE2Poses:
    def __init__(self, t, R):
        self._single = False

        if t.ndim not in [1, 2] or t.shape[-1] != 2:
            raise ValueError("Expected `t` to have shape (2,) or"
                             "(N x 2), got {}.".format(t.shape))

        if t.shape == (2,):
            t = t[None, :]
            self._single = True
            try:
                if not R.single: raise ValueError("Different number of translations 1 and rotations {}.".format(len(R)))
            except:
                if not R._single: raise ValueError("Different number of translations 1 and rotations {}.".format(len(R)))
             
        elif len(t) == 1:
            self._single = True
        else:
            if len(t) != len(R):
                raise ValueError("Differing number of translations {}"
                                 "and rotations {}".format(
                                     len(t), len(R)))
        self._t = t
        self._R = R
        try:
            if not R.single: self.len = len(R)
            else: self.len = 1
        except:
            if not R._single: self.len = len(R)
            else: self.len = 1

    def __getitem__(self, indexer):
        return self.__class__(self.t()[indexer], self.R()[indexer])

    def __len__(self):
        return self.len

    def __mul__(self, other):
        """
        Performs element-wise pose composition.
        TO DO: Broadcasting
        """
        if not(self.len == 1 or other.len == 1 or
               self.len == other.len):
            raise ValueError("Expected equal number of transformations"
                             "in both or a single transformation in"
                             "either object, got {} transformations"
                             "in first and {} transformations in "
                             "second object.".format(
                                len(self), len(other)))
        return self.__class__(self.R().apply2(other.t()) + self.t(),
                              self.R() * other.R())

    def __truediv__(self, other):
        """
        Computes relative pose, similar to MATLAB convention
        (x = A \ b for Ax = b). Example: T1 / T2 = T1.inv() * T2
        TO DO: Broadcasting
        """
        if not(self.len == 1 or other.len == 1 or self.len == other.len):
            raise ValueError("Expected equal number of transformations"
                                "in both or a single transformation in"
                                "either object, got {} transformations"
                                "in first and {} transformations in "
                                "second object.".format(
                                    len(self), len(other)))
        R1_inv = self.R().inv()
        t_new = R1_inv.apply2(other.t() - self.t())
        return self.__class__(t_new, R1_inv * other.R())

    def t(self):
        return self._t[0] if self._single else self._t

    def R(self):
        return self._R

    def inv(self):
        R_inv = self.R().inv()
        t_new = -R_inv.apply2(self.t())
        return SE2Poses(t_new, R_inv)

    def components(self):
        """
        Return translational and rotational components of pose
        separately. Quaternion form for rotations.
        """
        return self.t(), self.R()

    def repeat(self, N):
        t = self.t()
        q = self.R().as_quat()
        if len(self) == 1:
            t = np.expand_dims(t, 0)
            q = np.expand_dims(q, 0)
        t = np.repeat(t, N, axis=0)
        q = np.repeat(q, N, axis=0)
        return SE2Poses(t, Rotation2.from_quat(q))
        
    def as_numpy(self):
        """
        Return a pose array
        """
        return np.concatenate((self.t(), 
                               self.R().as_rotvec()), 
                               axis=-1)
        
        


def metric(p1, p2, w):
    """
    Computes metric on the cartesian product space
    representation of SE(3).

    Args:
        p1 (SE2Poses) : set of poses
        p2 (SE2Poses) : set of poses (same size as p1)
        w (float > 0) : weight for attitude component
    """
    if not(len(p1) == 1 or len(p2) == 1 or len(p1) == len(p2)):
        raise ValueError("Expected equal number of transformations in"
                            "both or a single transformation in either"
                            "object, got {} transformations in first"
                            "and {} transformations in second object."
                            .format(len(p1), len(p2)))
    if w < 0:
        raise ValueError("Weight must be non-negative, currently {}".
                         format(w))
    p_rel = p1 / p2
    t_dist = np.linalg.norm(p_rel.t(), axis=-1)
    R_dist = p_rel.R().magnitude()
    return t_dist + w * R_dist 


def combine(listOfPoses):
    """
    combines a list of SE2 objects into a single SE2 object
    """
    tList = []
    qList = []
    for pose in listOfPoses:
        if len(pose) == 1:
            t_temp = np.expand_dims(pose.t(), 0)
            q_temp = np.expand_dims(pose.R().as_quat(), 0)
        else:
            t_temp = pose.t()
            q_temp = pose.R().as_quat()
        tList.append(t_temp)
        qList.append(q_temp)
    tList = np.concatenate(tList, axis=0)
    qList = np.concatenate(qList, axis=0)
    return SE2Poses(tList, Rotation2.from_quat(qList))


def expSE2(twist):
    """
    Applies exponential map to twist vectors
    Args
        twist: N x 3 matrix or 3D vector containing se(2) element(s)
    Returns
        SE2Poses object with equivalent SE2 transforms
    """
    u = twist[..., :2]
    w = twist[..., 2:]

    R = Rotation2.from_euler('z', w.squeeze())
    theta = R.magnitude()
    what = hatOp(w)  # skew symmetric form
    with np.errstate(divide='ignore'):
        B = (1 - np.cos(theta)) / theta ** 2
        C = (theta - np.sin(theta)) / theta ** 3
    if len(twist.shape) == 2:
        B[np.abs(theta) < 1e-3] = 0.5  # limit for theta -> 0
        C[np.abs(theta) < 1e-3] = 1. / 6  # limit for theta -> 0
        V = np.eye(2)[np.newaxis, ...] + B[:, np.newaxis, np.newaxis]\
            * what + C[:, np.newaxis, np.newaxis] * what @ what
        V = V.squeeze()
    else:
        if np.abs(theta) < 1e-3:
            B = 0.5  # limit for theta -> 0
            C = 1. / 6  # limit for theta -> 0
        V = np.eye(2) + B * what + C * what @ what
    t = V @ u[..., np.newaxis]
    return SE2Poses(t.squeeze(), R)


def logSE2(T):
    """
    Applies inverse exponential map to SE2 elements
    Args
        T: SE2Poses element, may have 1 or more (N) poses
    Returns
        Nx3 matrix or 3D vector representing twists
    """
    R = T.R()
    t = T.t()

    theta = R.magnitude()
    with np.errstate(divide='ignore', invalid='ignore'):
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta ** 2
        sq_coeff = 1 / theta ** 2 * (1 - A / (2 * B))
        
    what = hatOp(R.as_rotvec())

    if not T._single:
        A[np.abs(theta) < 1e-3] = 1.  # limit for theta -> 0
        B[np.abs(theta) < 1e-3] = 0.5  # limit for theta -> 0
        sq_coeff[np.abs(theta) < 1e-3] = 1. / 12
        Vinv = np.eye(2)[np.newaxis, ...] - 0.5 *\
            what + sq_coeff[:, np.newaxis, np.newaxis] * what @ what
    else:
        if np.abs(theta) < 1e-3:
            A = 1.  # limit for theta -> 0
            B = 0.5  # limit for theta -> 0
            sq_coeff = 1. / 12
        Vinv = np.eye(2) - 0.5 * what + sq_coeff * what @ what
    u = Vinv @ t[..., np.newaxis]
    
    return np.concatenate((u.squeeze(), R.as_rotvec()), axis=-1)


def hatOp(vec):
    """
    Turns N vector into Nx2x2 skew skymmetric representation.
    Works for single 3D vector also.
    """
    if vec.size > 1:
        mat = np.zeros((vec.shape[0], 2, 2))
    else:
        mat = np.zeros((2, 2))
    mat[..., 1, 0] = vec[...].squeeze()
    
    if vec.size > 1:
        mat = mat - np.transpose(mat, (0, 2, 1))
    else:
        mat = mat - mat.transpose()
    return mat
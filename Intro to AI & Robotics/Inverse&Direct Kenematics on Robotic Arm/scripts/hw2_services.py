#!/usr/bin/env python

import rospy
import tf
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, TriggerResponse
import tf.transformations
from matplotlib import pyplot as plt

class GeometricServices(object):
    def __init__(self):
        self.current_joints = None

        # Subscribe to /joint_states topic
        rospy.Subscriber("/joint_states", JointState, self.joints_callback)

        # Create service callback to tf translations
        rospy.Service('get_tf_ee', Trigger, self.get_tf_ee_callback)
        self.tf_listener = tf.TransformListener()

        # init your own kinematic chain offsets
        self.a_i = [0, 0.8, 0, 0, 0, 0]  # TODO3
        self.alpha_i = [-np.pi/2, 0, np.pi/2, -np.pi/2, np.pi/2, 0]
        self.d_i = [0.8, 0, 0, 1.1, 0.05, 0.6]
        self.nue_i = None  # TODO3

        # Create service callback to ee pose
        self.direct_translation = rospy.Service('get_ee_pose', Trigger, self.get_ee_pose_callback)

    def joints_callback(self, msg):
        self.current_joints = msg.position

    def get_tf_ee_callback(self, reqt):
        try:
            trans, rot = self.tf_listener.lookupTransform('base_link', 'end-effector-link', rospy.Time(0))
            message = 'translation {}, rotation {}'.format(trans, rot)
            return TriggerResponse(success=True, message=message)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return TriggerResponse(success=False, message='Failed, is TF running?')

    def get_ee_pose_callback(self, reqt):
        _, translation, rotation = self._get_ee_pose(self.current_joints)
        message = 'translation {} rotation {}'.format(translation, rotation)
        return TriggerResponse(success=True, message=message)

    def _get_ee_pose(self, joints):

        self.nue_i = [-np.pi / 2 + joints[0], -np.pi / 2 + joints[1], np.pi / 2 + joints[2],
                      joints[3], joints[4], np.pi / 2 + joints[5]]

        T_1 = np.eye(4)
        for a, alpha, d, nue in zip(self.a_i, self.alpha_i, self.d_i, self.nue_i):
            T = self._generate_homogeneous_transformation(a, alpha, d, nue)
            T_1 = np.dot(T_1, T)
        Tb = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.1],
                        [0, 0, 0, 1]])

        Tout = np.dot(Tb, T_1)

        translation = Tout[:3, 3]
        rotation_matrix = Tout[:3, :3]
        quaternion = self._rotation_to_quaternion(rotation_matrix)

        return Tout, translation, quaternion

    @staticmethod
    def _generate_homogeneous_transformation(a, alpha, d, nue):
        T = np.array([[np.cos(nue), -np.sin(nue) * np.cos(alpha), np.sin(nue) * np.sin(alpha), a * np.cos(nue)],
                      [np.sin(nue), np.cos(nue) * np.cos(alpha), -np.cos(nue) * np.sin(alpha), a * np.sin(nue)],
                      [0, np.sin(alpha), np.cos(alpha), d],
                      [0, 0, 0, 1]])
        return T

    @staticmethod
    def _rotation_to_quaternion(r):
        x = 0.5 * np.sign(r[2, 1] - r[1, 2]) * np.sqrt(r[0, 0] - r[1, 1] - r[2, 2] + 1)
        y = 0.5 * np.sign(r[0, 2] - r[2, 0]) * np.sqrt(r[1, 1] - r[2, 2] - r[0, 0] + 1)
        z = 0.5 * np.sign(r[1, 0] - r[0, 1]) * np.sqrt(r[2, 2] - r[0, 0] - r[1, 1] + 1)
        real = 0.5 * np.sqrt(r[0, 0] + r[1, 1] + r[2, 2] + 1)
        return np.array([x, y, z, real])

    def get_geometric_jacobian(self, joints):
        # TODO8
        _, ee_pos, _ = self._get_ee_pose(joints)
        j = np.zeros((6, len(joints)))
        self.nue_i = [-np.pi/2+joints[0],-np.pi/2+joints[1],np.pi/2+joints[2],joints[3],joints[4],np.pi/2+joints[5]]
        T = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0.1],
                                [0, 0, 0, 1]])

        z_prev = T[:3, 2]
        p_prev = T[:3, 3]

        for i in range(len(joints)):
            T_current = self._generate_homogeneous_transformation(self.a_i[i], self.alpha_i[i], self.d_i[i],
                                                                  self.nue_i[i])
            T = np.dot(T, T_current)

            z_curr = T[:3, 2]
            p_curr = T[:3, 3]

            j_pi = np.cross(z_prev, np.subtract(ee_pos, p_prev))
            j_oi = z_prev

            j[:3, i] = j_pi
            j[3:, i] = j_oi

            z_prev = z_curr
            p_prev = p_curr
        return j

    def get_analytical_jacobian(self, joints):
        geometric_j = self.get_geometric_jacobian(joints)
        j_p = geometric_j[:3, :]
        j_o = geometric_j[3:, :]

        j_a = np.zeros((6, len(joints)))
        _, _, ee_q = self._get_ee_pose(joints)
        phi, theta, psi = convert_quaternion_to_zyz(ee_q)

        T = np.array([
            [0, -np.sin(phi), np.cos(phi) * np.sin(theta)],
            [0, np.cos(phi), np.sin(phi) * np.sin(theta)],
            [1, 0, np.cos(theta)]
        ])
        T_inv = np.linalg.pinv(T)

        j_a[:3, :] = j_p
        j_a[3:, :] = np.dot(T_inv, j_o)

        return j_a

    def compute_inverse_kinematics(self, end_pose, max_iterations, error_threshold, time_step, initial_joints, k=1.):
        # Initialization of tracking lists and setup variables
        iteration_steps = []
        pose_errors = []
        current_configuration = initial_joints
        error_magnitude = error_threshold + 1

        for step_index in range(max_iterations):
            if error_magnitude <= error_threshold:
                break

            _, translation_vector, quaternion_orientation = self._get_ee_pose(current_configuration)
            euler_angles = convert_quaternion_to_zyz(quaternion_orientation)
            full_current_pose = np.concatenate([translation_vector, euler_angles])
            deviation = full_current_pose - end_pose
            error_magnitude = np.linalg.norm(deviation, ord=2)

            print("Iteration Num {}, Current Error: {}".format(step_index + 1, error_magnitude))

            analytical_jacobian = self.get_analytical_jacobian(current_configuration)
            jacobian_transpose = np.transpose(analytical_jacobian)
            jacobian_product = np.dot(analytical_jacobian, jacobian_transpose)
            damped_inversion = np.linalg.inv(jacobian_product + k ** 2 * np.eye(jacobian_product.shape[0]))
            damped_jacobian_pseudo_inverse = np.dot(jacobian_transpose, damped_inversion)

            joint_update = np.dot(damped_jacobian_pseudo_inverse, deviation)
            current_configuration -= joint_update * time_step
            self._normalize_joints(current_configuration)
            iteration_steps.append(step_index + 1)
            pose_errors.append(deviation)

        pose_errors = np.array(pose_errors)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot for position errors
        ax1.plot(iteration_steps, pose_errors[:, :3], marker='o')
        ax1.set_title("Position Error Trends Over Iterations")
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Position Error (units)")
        ax1.grid(True)
        ax1.legend(['X-axis', 'Y-axis', 'Z-axis'])

        # Plot for orientation errors
        ax2.plot(iteration_steps, pose_errors[:, 3:], marker='x')
        ax2.set_title("Orientation Error Trends Over Iterations")
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Orientation Error (radians)")
        ax2.grid(True)
        ax2.legend(['Rotation around X', 'Rotation around Y', 'Rotation around Z'])

        plt.tight_layout()
        plt.show()

        final_pos_error = pose_errors[-1, :3]
        final_orient_error = pose_errors[-1, 3:]
        print("Final Position Error:{}, Final Orientation Error: {}".format(final_pos_error, final_orient_error))
        return current_configuration


    @staticmethod
    def _normalize_joints(joints):
        res = [j for j in joints]
        for i in range(len(res)):
            res[i] = res[i] + np.pi
            res[i] = res[i] % (2 * np.pi)
            res[i] = res[i] - np.pi
        return np.array(res)


def convert_quaternion_to_zyz(q):
    x, y, z, w = q
    x2, y2, z2 = x * x, y * y, z * z

    r11 = 1 - 2 * (y2 + z2)
    r12 = 2 * (x * y - z * w)
    r13 = 2 * (x * z + y * w)

    r21 = 2 * (x * y + z * w)
    r22 = 1 - 2 * (x2 + z2)
    r23 = 2 * (y * z - x * w)

    r31 = 2 * (x * z - y * w)
    r32 = 2 * (y * z + x * w)
    r33 = 1 - 2 * (x2 + y2)

    rotation_matrix = np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])

    phi = np.arctan2(rotation_matrix[1, 2], rotation_matrix[0, 2])
    theta = np.arctan2(np.sqrt(rotation_matrix[0, 2] ** 2 + rotation_matrix[1, 2] ** 2), rotation_matrix[2, 2])
    psi = np.arctan2(rotation_matrix[2, 1], -rotation_matrix[2, 0])

    return [phi, theta, psi]


def solve_ik(geometric_services):
    end_position = [-0.770, 1.562, 1.050]
    end_zyz = convert_quaternion_to_zyz([0.392, 0.830, 0.337, -0.207])
    end_pose = np.concatenate((end_position, end_zyz), axis=0)
    result = gs.compute_inverse_kinematics(end_pose, max_iterations=10000, error_threshold=0.001, time_step=0.001,
                                           initial_joints=[0.1] * 6)
    print('ik solution {}'.format(result))



if __name__ == '__main__':
    rospy.init_node('hw2_services_node')
    gs = GeometricServices()
    solve_ik(gs)
    q = [0.392, 0.830, 0.337, -0.207]
    phi , theta , psi = convert_quaternion_to_zyz(q)
    print([phi, theta, psi])
    rospy.spin()

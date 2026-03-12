import numpy as np
import scipy
np.set_printoptions(precision=3, suppress=True)
from matplotlib import pyplot as plt

def rotation_x(angle):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def rotation_y(angle):
    return np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def rotation_z(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def translation(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

class InverseKinematics():

    def __init__(self):
        self.joint_positions = None
        self.joint_velocities = None
        self.target_joint_positions = None
        self.counter = 0

        # Trotting gait positions
        # Triangle: base width = 0.10m, height = 0.09m
        # Bottom center at [0, 0, -0.14], y=0 for all
        # From the diagram:
        #   Touch Down is at the RIGHT side of the base (forward in x)
        #   Lift Off is at the LEFT side of the base (backward in x)
        #   Stand 1, 2, 3 are evenly spaced along the bottom from right to left
        #   Mid Swing is at the top of the triangle (above center)
        #
        # Base width = 0.10, so half = 0.05
        # Bottom center = [0, 0, -0.14]
        # Touch Down (right) = [0.05, 0, -0.14]
        # Lift Off (left)    = [-0.05, 0, -0.14]
        # Stand 1 = 1/4 from right = [0.05 - 0.10/4, 0, -0.14] = [0.025, 0, -0.14]
        # Stand 2 = center = [0, 0, -0.14]
        # Stand 3 = 3/4 from right = [0.05 - 3*0.10/4, 0, -0.14] = [-0.025, 0, -0.14]
        # Mid Swing = top = [0, 0, -0.14 + 0.09] = [0, 0, -0.05]

        touch_down_position = np.array([0.05, 0, -0.14])
        stand_position_1    = np.array([0.025, 0, -0.14])
        stand_position_2    = np.array([0.0, 0, -0.14])
        stand_position_3    = np.array([-0.025, 0, -0.14])
        liftoff_position    = np.array([-0.05, 0, -0.14])
        mid_swing_position  = np.array([0.0, 0, -0.05])
        
        ## trotting
        # In a trot gait, diagonal legs are in phase:
        #   RF and LB start at touch_down (beginning of stance phase)
        #   LF and RB start at stand_position_3 (end of stance, about to swing)
        # The sequence for a full cycle is:
        #   touch_down -> stand_1 -> stand_2 -> stand_3 -> liftoff -> mid_swing -> (back to touch_down)
        
        rf_ee_offset = np.array([0.06, -0.09, 0])
        rf_ee_triangle_positions = np.array([
            touch_down_position,
            stand_position_1,
            stand_position_2,
            stand_position_3,
            liftoff_position,
            mid_swing_position,
        ]) + rf_ee_offset
        
        lf_ee_offset = np.array([0.06, 0.09, 0])
        lf_ee_triangle_positions = np.array([
            stand_position_3,
            liftoff_position,
            mid_swing_position,
            touch_down_position,
            stand_position_1,
            stand_position_2,
        ]) + lf_ee_offset
        
        rb_ee_offset = np.array([-0.11, -0.09, 0])
        rb_ee_triangle_positions = np.array([
            stand_position_3,
            liftoff_position,
            mid_swing_position,
            touch_down_position,
            stand_position_1,
            stand_position_2,
        ]) + rb_ee_offset
        
        lb_ee_offset = np.array([-0.11, 0.09, 0])
        lb_ee_triangle_positions = np.array([
            touch_down_position,
            stand_position_1,
            stand_position_2,
            stand_position_3,
            liftoff_position,
            mid_swing_position,
        ]) + lb_ee_offset


        self.ee_triangle_positions = [rf_ee_triangle_positions, lf_ee_triangle_positions, rb_ee_triangle_positions, lb_ee_triangle_positions]
        self.fk_functions = [self.fr_leg_fk, self.fl_leg_fk, self.br_leg_fk, self.bl_leg_fk]

        ####### Uncomment when you want to generate the cache #######
        self.target_joint_positions_cache, self.target_ee_cache = self.cache_target_joint_positions()
        print(f'shape of target_joint_positions_cache: {self.target_joint_positions_cache.shape}')
        print(f'shape of target_ee_cache: {self.target_ee_cache.shape}')


    def fr_leg_fk(self, theta):
        T_RF_0_1 = translation(0.07500, -0.08350, 0) @ rotation_x(1.57080) @ rotation_z(theta[0])
        T_RF_1_2 = rotation_y(-1.57080) @ rotation_z(theta[1])
        T_RF_2_3 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta[2])
        T_RF_3_ee = translation(0.06231, -0.06216, 0.01800)
        T_RF_0_ee = T_RF_0_1 @ T_RF_1_2 @ T_RF_2_3 @ T_RF_3_ee
        return T_RF_0_ee[:3, 3]

    def fl_leg_fk(self, theta):
        T_LF_0_1 = translation(0.07500, 0.08350, 0) @ rotation_x(1.57080) @ rotation_z(-theta[0])
        T_LF_1_2 = rotation_y(-1.57080) @ rotation_z(theta[1])
        T_LF_2_3 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(-theta[2])
        T_LF_3_ee = translation(0.06231, -0.06216, -0.01800)
        T_LF_0_ee = T_LF_0_1 @ T_LF_1_2 @ T_LF_2_3 @ T_LF_3_ee
        return T_LF_0_ee[:3, 3]

    def br_leg_fk(self, theta):
        T_RB_0_1 = translation(-0.07500, -0.07250, 0) @ rotation_x(1.57080) @ rotation_z(theta[0])
        T_RB_1_2 = rotation_y(-1.57080) @ rotation_z(theta[1])
        T_RB_2_3 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(theta[2])
        T_RB_3_ee = translation(0.06231, -0.06216, 0.01800)
        T_RB_0_ee = T_RB_0_1 @ T_RB_1_2 @ T_RB_2_3 @ T_RB_3_ee
        return T_RB_0_ee[:3, 3]

    def bl_leg_fk(self, theta):
        T_LB_0_1 = translation(-0.07500, 0.07250, 0) @ rotation_x(1.57080) @ rotation_z(-theta[0])
        T_LB_1_2 = rotation_y(-1.57080) @ rotation_z(theta[1])
        T_LB_2_3 = translation(0, -0.04940, 0.06850) @ rotation_y(1.57080) @ rotation_z(-theta[2])
        T_LB_3_ee = translation(0.06231, -0.06216, -0.01800)
        T_LB_0_ee = T_LB_0_1 @ T_LB_1_2 @ T_LB_2_3 @ T_LB_3_ee
        return T_LB_0_ee[:3, 3]

    def forward_kinematics(self, theta):
        return np.concatenate([self.fk_functions[i](theta[3*i: 3*i+3]) for i in range(4)])
        
    def get_error_leg(self, theta, desired_position):
        """Calculate scalar error between FK result and desired position."""
        current_position = self.leg_forward_kinematics(theta)
        error = np.linalg.norm(current_position - desired_position)
        return error

    def inverse_kinematics_single_leg(self, target_ee, leg_index, initial_guess=[0, 0, 0]):
        self.leg_forward_kinematics = self.fk_functions[leg_index]
        result = scipy.optimize.minimize(
            self.get_error_leg,
            initial_guess,
            args=(target_ee,),
            method='SLSQP'
        )
        return result.x

    def interpolate_triangle(self, t, leg_index):
        """
        Interpolate along the triangle waypoints for a given leg.
        t is between 0 and 1, representing progress through the full cycle.
        """
        waypoints = self.ee_triangle_positions[leg_index]
        n = len(waypoints)  # 6 waypoints
        
        # Scale t to the total number of segments (n segments because we wrap around)
        t_scaled = t * n
        # Determine which segment we're on
        segment_index = int(t_scaled) % n
        # How far along this segment (0 to 1)
        segment_t = t_scaled - int(t_scaled)
        
        # Get the two waypoints to interpolate between
        p_start = waypoints[segment_index]
        p_end = waypoints[(segment_index + 1) % n]
        
        # Linear interpolation
        position = p_start + segment_t * (p_end - p_start)
        return position

    def cache_target_joint_positions(self):
        # Calculate and store the target joint positions for a cycle and all 4 legs
        target_joint_positions_cache = []
        target_ee_cache = []
        for leg_index in range(4):
            target_joint_positions_cache.append([])
            target_ee_cache.append([])
            target_joint_positions = [0] * 3
            for t in np.arange(0, 1, 0.02):
                target_ee = self.interpolate_triangle(t, leg_index)
                target_joint_positions = self.inverse_kinematics_single_leg(target_ee, leg_index, initial_guess=target_joint_positions)
                target_joint_positions_cache[leg_index].append(target_joint_positions)
                target_ee_cache[leg_index].append(target_ee)

        # (4, 50, 3) -> (50, 12)
        target_joint_positions_cache = np.concatenate(target_joint_positions_cache, axis=1)
        target_ee_cache = np.concatenate(target_ee_cache, axis=1)
        
        return target_joint_positions_cache, target_ee_cache

    def get_target_joint_positions(self):
        target_joint_positions = self.target_joint_positions_cache[self.counter]
        target_ee = self.target_ee_cache[self.counter]
        self.counter += 1
        if self.counter >= self.target_joint_positions_cache.shape[0]:
            self.counter = 0
        return target_ee, target_joint_positions




def main():

    # Create an instance of the IK class
    inverse_kinematics = InverseKinematics()

    # Test IK solver on right front leg
    target_ee_list = [
        [0.11, -0.09, -0.14],
        [0.10, -0.09, -0.14],
        [0.09, -0.09, -0.14],
        [0.08, -0.09, -0.14],
        [0.07, -0.09, -0.14],
        [0.06, -0.09, -0.14],
        [0.05, -0.09, -0.14],
        [0.04, -0.09, -0.14],
        [0.03, -0.09, -0.14],
        [0.02, -0.09, -0.14],
        [0.01, -0.09, -0.14],
    ]

    inverse_kinematics.leg_forward_kinematics = inverse_kinematics.fk_functions[0]
    result_ee_list = []
    theta_list = []
    for target_ee in target_ee_list:
        theta = inverse_kinematics.inverse_kinematics_single_leg(target_ee, leg_index=0, initial_guess=[0, 0, 0])
        result_ee = inverse_kinematics.leg_forward_kinematics(theta)
        result_ee_list.append(result_ee)
        theta_list.append(theta)
        print(f'Target: {target_ee}, Result: {result_ee}, Theta: {theta}')

    # Plot the EE results
    if len(result_ee_list) > 0:
        plt.figure()
        plt.plot(np.array(target_ee_list)[:,0],'k')
        plt.plot(np.array(result_ee_list)[:,0],'ro')
        plt.xlabel('Step')
        plt.ylabel('X (m)')
        plt.legend(['Target EE Position','Result EE Position'])
        plt.title('End Effector X position')
        plt.savefig('/Users/ahbabalmohammadsiddiquee/Downloads/lab3/ik_plot.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Plot joint angles
    if len(theta_list) > 0:
        theta_arr = np.array(theta_list)
        plt.figure()
        plt.plot(theta_arr[:, 0], label='theta_0 (hip)')
        plt.plot(theta_arr[:, 1], label='theta_1 (upper)')
        plt.plot(theta_arr[:, 2], label='theta_2 (lower)')
        plt.xlabel('Step')
        plt.ylabel('Angle (rad)')
        plt.legend()
        plt.title('Joint Angles from IK')
        plt.savefig('/Users/ahbabalmohammadsiddiquee/Downloads/lab3/joint_angles_plot.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Plot the cached trot gait path for front right foot (leg_index=0).
    if hasattr(inverse_kinematics, 'target_ee_cache') and len(inverse_kinematics.target_ee_cache) > 0:
        # target_ee_cache shape is (50, 12) — first 3 columns are RF leg
        x_list = []
        z_list = []
        for position in inverse_kinematics.target_ee_cache:
            x_list.append(position[0])  # RF x
            z_list.append(position[2])  # RF z
        plt.figure()
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.title('EE front right foot trot gait')
        plt.plot(x_list, z_list, 'b.-')
        plt.savefig('/Users/ahbabalmohammadsiddiquee/Downloads/lab3/trot_gait_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also plot front left foot (columns 3:6)
        x_list_lf = []
        z_list_lf = []
        for position in inverse_kinematics.target_ee_cache:
            x_list_lf.append(position[3])  # LF x
            z_list_lf.append(position[5])  # LF z
        plt.figure()
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.title('EE front left foot trot gait')
        plt.plot(x_list_lf, z_list_lf, 'r.-')
        plt.savefig('/Users/ahbabalmohammadsiddiquee/Downloads/lab3/trot_gait_lf_plot.png', dpi=150, bbox_inches='tight')
        plt.close()



if __name__ == '__main__':
    main()

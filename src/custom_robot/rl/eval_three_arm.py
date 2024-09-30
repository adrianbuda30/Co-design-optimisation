import rospy
import math
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from tf.transformations import quaternion_from_euler

def create_arm_marker(arm_id, start_point, end_point, arm_diameter):
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.type = marker.CYLINDER
    marker.action = marker.ADD
    marker.ns = "robot_arms"
    marker.id = arm_id
    marker.scale.x = arm_diameter
    marker.scale.y = arm_diameter
    marker.scale.z = math.sqrt((end_point.x - start_point.x) ** 2 + (end_point.y - start_point.y) ** 2)
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0

    # Cylinder's pose is at the midpoint between start_point and end_point
    marker.pose.position.x = (start_point.x + end_point.x) / 2
    marker.pose.position.y = (start_point.y + end_point.y) / 2
    marker.pose.position.z = 0

    # Compute the angle to rotate the cylinder
    angle = math.atan2(end_point.y - start_point.y, end_point.x - start_point.x)
    q = quaternion_from_euler(0, 0, 90)  # Convert euler to quaternion

    marker.pose.orientation.x = q[0]
    marker.pose.orientation.y = q[1]
    marker.pose.orientation.z = q[2]
    marker.pose.orientation.w = q[3]

    return marker

def main():
    rospy.init_node('robot_arm_visualization')

    # Publisher for arm markers
    pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
    rate = rospy.Rate(30)  # 30 Hz for smoother animation

    # Define the arm lengths
    arm_lengths = [1.0, 1.0, 1.0]  # Lengths of the arms

    # Time variable
    t = 0

    while not rospy.is_shutdown():
        # Update joint angles using a wave function, e.g., sine wave
        joint_angles = [math.sin(t), math.sin(t + math.pi/2), math.sin(t + math.pi)]

        # Calculate the position of each joint
        joint_positions = [Point(0, 0, 0)]  # Base position

        for i, (length, angle) in enumerate(zip(arm_lengths, joint_angles), start=1):
            prev_point = joint_positions[-1]
            new_point = Point()
            new_point.x = prev_point.x + length * math.cos(angle)
            new_point.y = prev_point.y + length * math.sin(angle)
            joint_positions.append(new_point)

            # Create and publish a marker for each arm
            arm_marker = create_arm_marker(i, prev_point, new_point, 0.05)
            pub.publish(arm_marker)

        # Increment time
        t += 0.1

        # Sleep for the remainder of the loop
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

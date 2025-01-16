
from FrankaClient import *


def main():

    client = FrankaClient(
    server_ip='127.0.0.1',
)

    home = [-0.946566641330719, -1.316757321357727, 1.2592488527297974, -2.814577102661133, 1.2752666473388672, 1.7911940813064575, 0.052171170711517334]
    client.move_to_joint_positions(home, 10)
        

if __name__ == "__main__":
    main()

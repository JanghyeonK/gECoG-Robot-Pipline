#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Go2 Monkey Control - ZMQ Version
Remote control via ZMQ messages
"""

import time
import sys
import signal
import threading
import math
import numpy as np
import os
import zmq

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
    print("✓ SDK loaded")
except ImportError as e:
    print(f"✗ SDK error: {e}")
    sys.exit(1)

# Global wave timer
_wave_t0 = None

class Go2MonkeyZMQController:
    def __init__(self):
        self.nic = "enxb0386cf170af"
        self.sport_client = None
        self.robot_state_client = None

        # ZMQ settings
        self.zmq_address = "tcp://192.168.43.226:5555"
        self.context = None
        self.socket_sub = None

        # Speed settings
        self.walk_speed = 0.15  # m/s
        self.turn_speed = 0.25  # rad/s

        # State management
        self.running = True
        self.current_action = None
        self.action_thread = None
        self.stop_action = False
        self.is_low_level = False

        # Low-level components
        self.low_cmd_pub = None
        self.low_state_sub = None
        self.low_state = None
        self.crc = None
        self.low_cmd = None

        # Action name mapping
        self.action_names = {
            0: "Walk forward",
            1: "Hello gesture",
            2: "Walk forward",
            3: "Turn left",
            4: "Sit",
            5: "Turn left",
            6: "Turn right",
            7: "Stand"
        }

        print("="*50)
        print("GO2 MONKEY CONTROLLER - ZMQ VERSION")
        print(f"ZMQ Subscriber: {self.zmq_address}")
        print("="*50)

    def init_zmq(self):
        """Initialize ZMQ Subscriber."""
        try:
            print(f"\nConnecting ZMQ: {self.zmq_address}")
            self.context = zmq.Context()
            self.socket_sub = self.context.socket(zmq.SUB)
            self.socket_sub.connect(self.zmq_address)
            self.socket_sub.setsockopt_string(zmq.SUBSCRIBE, "")
            self.socket_sub.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
            print(f"✓ ZMQ Subscriber connected")
            return True
        except Exception as e:
            print(f"✗ ZMQ connection failed: {e}")
            return False

    def init_clients(self):
        """Initialize SDK clients."""
        print("\nInitializing...")

        # Initialize DDS
        ChannelFactoryInitialize(0, self.nic)
        print("✓ DDS connected")

        # Initialize RobotStateClient (required for ServiceSwitch)
        try:
            self.robot_state_client = RobotStateClient()
            self.robot_state_client.SetTimeout(10.0)
            self.robot_state_client.Init()
            print("✓ RobotStateClient ready (ServiceSwitch)")
        except Exception as e:
            print(f"✗ RobotStateClient failed: {e}")
            print("  Low-level control unavailable")
            return False

        # Initialize SportClient
        try:
            self.sport_client = SportClient()
            self.sport_client.SetTimeout(5.0)
            self.sport_client.Init()
            print("✓ SportClient ready")
        except Exception as e:
            print(f"✗ SportClient failed: {e}")
            return False

        return True

    def prepare_robot(self):
        """Prepare robot to stand."""
        print("\nPreparing robot...")

        try:
            self.sport_client.StandUp()
            time.sleep(1.5)

            self.sport_client.BalanceStand()
            time.sleep(0.3)

            print("✓ Robot ready")
            return True

        except Exception as e:
            print(f"✗ Preparation failed: {e}")
            return False

    def init_low_level(self):
        """Initialize low-level control."""
        print("  Initializing low-level...")

        self.low_cmd_pub = None
        self.low_state_sub = None
        self.low_state = None
        self.low_cmd = None

        self.low_cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_pub.Init()

        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_sub.Init(lambda msg: setattr(self, 'low_state', msg))

        self.crc = CRC()
        self.low_cmd = unitree_go_msg_dds__LowCmd_()

        for i in range(12):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0

        time.sleep(0.3)
        print("  ✓ Low-level initialized")

    def send_low_cmd(self, q, dq, kp, kd, tau):
        """Send low-level motor command."""
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = float(q[i])
            self.low_cmd.motor_cmd[i].dq = float(dq[i])
            self.low_cmd.motor_cmd[i].kp = float(kp[i])
            self.low_cmd.motor_cmd[i].kd = float(kd[i])
            self.low_cmd.motor_cmd[i].tau = float(tau[i])

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.low_cmd_pub.Write(self.low_cmd)

    def get_current_joints(self):
        """Get current joint positions."""
        if self.low_state:
            return [self.low_state.motor_state[i].q for i in range(12)]
        # Default prone pose if low_state unavailable
        return [0.0, 1.36, -2.65,    # FR
                0.0, 1.36, -2.65,    # FL
                -0.2, 1.36, -2.65,   # RR
                0.2, 1.36, -2.65]    # RL

    def action_0_walk(self):
        """Action 0: Walk forward."""
        print("\n[Action 0] Walking...")
        while not self.stop_action:
            try:
                self.sport_client.Move(self.walk_speed, 0.0, 0.0)
                time.sleep(0.1)
            except:
                break

    def action_1_eat(self):
        """Action 1: Wave gesture (Low-level)."""
        print("\n[Action 1] Wave gesture (Low-level)...")

        # === Phase 1: sport_mode OFF ===
        print("\n[Phase 1] sport_mode service OFF...")
        try:
            self.sport_client.StandDown()
            time.sleep(1.0)

            ret = self.robot_state_client.ServiceSwitch("sport_mode", 0)
            if ret == 0:
                print("✓ sport_mode OFF success")
            else:
                print(f"⚠ sport_mode OFF failed: ret={ret}")
                return

            time.sleep(1.2)

        except Exception as e:
            print(f"✗ ServiceSwitch OFF failed: {e}")
            return

        self.init_low_level()
        self.is_low_level = True

        # === Phase 2: Low-level wave gesture ===
        print("\n[Phase 2] Low-level wave gesture...")

        sitting_q = np.array([
            -0.20, 1.57, -2.77,   # FR
            0.20, 1.57, -2.77,    # FL
            -0.20, 2.77, -2.77,   # RR
            0.20, 2.77, -2.77     # RL
        ], dtype=np.float32)

        q_now = np.array(self.get_current_joints(), dtype=np.float32)

        for step in range(100):
            if self.stop_action:
                break
            a = (step + 1.0) / 100
            q_cmd = (1 - a) * q_now + a * sitting_q
            self.send_low_cmd(q_cmd,
                            np.zeros(12),
                            np.full(12, 60.0, np.float32),
                            np.full(12, 5.0, np.float32),
                            np.zeros(12))
            time.sleep(0.01)

        print("Waving...")
        global _wave_t0
        _wave_t0 = time.perf_counter()

        for i in range(300):
            if self.stop_action:
                break

            wave_q = sitting_q.copy()
            wave_q[2] = -2.77 + 0.5 * math.sin(i * 0.05)

            self.send_low_cmd(wave_q,
                            np.zeros(12),
                            np.full(12, 60.0, np.float32),
                            np.full(12, 5.0, np.float32),
                            np.zeros(12))
            time.sleep(0.01)

        print("Returning to prone pose...")
        lying_q = np.array([
            0.0, 1.36, -2.65,    # FR
            0.0, 1.36, -2.65,    # FL
            -0.2, 1.36, -2.65,   # RR
            0.2, 1.36, -2.65     # RL
        ], dtype=np.float32)

        q_now = np.array(self.get_current_joints(), dtype=np.float32)

        for step in range(100):
            a = (step + 1.0) / 100
            q_cmd = (1 - a) * q_now + a * lying_q
            self.send_low_cmd(q_cmd,
                            np.zeros(12),
                            np.full(12, 60.0, np.float32),
                            np.full(12, 5.0, np.float32),
                            np.zeros(12))
            time.sleep(0.01)

        print("Releasing low-level control...")
        for i in range(12):
            self.low_cmd.motor_cmd[i].mode = 0x00  # Damping
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 2.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.low_cmd_pub.Write(self.low_cmd)
        time.sleep(0.3)

        # === Phase 3: sport_mode ON ===
        print("\n[Phase 3] sport_mode service ON...")
        try:
            ret = self.robot_state_client.ServiceSwitch("sport_mode", 1)
            if ret == 0:
                print("✓ sport_mode ON success")
            else:
                print(f"⚠ sport_mode ON failed: ret={ret}")

            time.sleep(1.0)

            print("Standing up...")
            self.sport_client.StandUp()
            time.sleep(1.5)
            self.sport_client.BalanceStand()

            print("✓ High-level mode restored")

        except Exception as e:
            print(f"✗ ServiceSwitch ON failed: {e}")

        self.is_low_level = False
        _wave_t0 = None

    def action_2_walk(self):
        """Action 2: Walk forward."""
        print("\n[Action 2] Walking...")
        while not self.stop_action:
            try:
                self.sport_client.Move(self.walk_speed, 0.0, 0.0)
                time.sleep(0.1)
            except:
                break

    def action_3_turn_left(self):
        """Action 3: Turn left."""
        print("\n[Action 3] Turning left...")
        while not self.stop_action:
            try:
                self.sport_client.Move(0.0, 0.0, self.turn_speed)
                time.sleep(0.1)
            except:
                break

    def action_4_sit(self):
        """Action 4: Sit down."""
        print("\n[Action 4] Sitting...")
        try:
            self.sport_client.BodyHeight(-0.1)

            while not self.stop_action:
                time.sleep(0.1)

            print("Returning to position...")
            self.sport_client.BodyHeight(0.0)
            self.sport_client.BalanceStand()

        except Exception as e:
            print(f"Sit error: {e}")

    def action_5_turn_left(self):
        """Action 5: Turn left."""
        print("\n[Action 5] Turning left...")
        while not self.stop_action:
            try:
                self.sport_client.Move(0.0, 0.0, self.turn_speed)
                time.sleep(0.1)
            except:
                break

    def action_6_turn_right(self):
        """Action 6: Turn right."""
        print("\n[Action 6] Turning right...")
        while not self.stop_action:
            try:
                self.sport_client.Move(0.0, 0.0, -self.turn_speed)
                time.sleep(0.1)
            except:
                break

    def action_7_stand(self):
        """Action 7: Stand up."""
        print("\n[Action 7] Standing up...")
        try:
            self.sport_client.StandUp()
            time.sleep(0.8)
            self.sport_client.BalanceStand()

            while not self.stop_action:
                time.sleep(0.1)

        except Exception as e:
            print(f"Stand error: {e}")

    def action_stop(self):
        """Stop action (class 9 or low probability)."""
        print("\n[Stop] Stopping action")
        self.stop_current_action()

    def stop_current_action(self):
        """Stop current action."""
        self.stop_action = True

        if self.action_thread and self.action_thread.is_alive():
            print("  Waiting for action to finish...")
            self.action_thread.join(timeout=6.0)

        try:
            self.sport_client.StopMove()
            time.sleep(0.1)
            self.sport_client.BalanceStand()
            print("  ✓ Safely stopped")
        except Exception as e:
            print(f"  Stop error: {e}")

    def execute_action(self, action_id):
        """Execute action by ID."""
        if self.action_thread and self.action_thread.is_alive():
            self.stop_current_action()

        actions = {
            0: self.action_0_walk,
            1: self.action_1_eat,
            2: self.action_2_walk,
            3: self.action_3_turn_left,
            4: self.action_4_sit,
            5: self.action_5_turn_left,
            6: self.action_6_turn_right,
            7: self.action_7_stand,
            9: self.action_stop,
        }

        if action_id in actions:
            self.stop_action = False
            self.current_action = action_id

            if action_id == 9:
                self.action_stop()
            else:
                self.action_thread = threading.Thread(target=actions[action_id])
                self.action_thread.start()

    def process_zmq_message(self, msg):
        """Process ZMQ message."""
        try:
            # Parse message: "pred_idx,prob,name"
            parts = msg.split(',', 2)
            if len(parts) != 3:
                print(f"⚠ Invalid message format: {msg}")
                return

            pred_idx = int(parts[0])
            prob = float(parts[1])
            name = parts[2]

            print(f"\n[Received] Class: {pred_idx} ({name}), Prob: {prob:.3f}")

            if prob < 0.5:
                print(f"  ⚠ Low probability ({prob:.3f} < 0.5) -> Stop")
                self.execute_action(9)
                return

            if 0 <= pred_idx <= 7:
                action_name = self.action_names.get(pred_idx, "Unknown")
                print(f"  -> Execute: {action_name}")
                self.execute_action(pred_idx)
            elif pred_idx == 9:
                print("  -> Stop")
                self.execute_action(9)
            else:
                print(f"  ⚠ Unknown class: {pred_idx}")

        except Exception as e:
            print(f"✗ Message processing error: {e}")
            print(f"  Original message: {msg}")

    def run(self):
        """Main loop."""
        print("\n" + "="*50)
        print("ZMQ Control Mode")
        print("="*50)
        print("\nAction mapping:")
        for idx, name in self.action_names.items():
            print(f"  {idx}: {name}")
        print("  9: Stop")
        print("\nAuto stop when probability < 0.5")
        print("="*50)

        if not self.init_clients():
            print("SDK initialization failed")
            return

        if not self.prepare_robot():
            print("Robot preparation failed")
            return

        if not self.init_zmq():
            print("ZMQ initialization failed")
            return

        print("\nReady! Waiting for ZMQ messages...")

        try:
            while self.running:
                try:
                    msg = self.socket_sub.recv_string()
                    self.process_zmq_message(msg)

                except zmq.Again:
                    pass

                except KeyboardInterrupt:
                    print("\n\n[Exit] Ctrl+C detected")
                    break

                except Exception as e:
                    print(f"✗ Receive error: {e}")

        finally:
            print("\nShutting down...")
            self.stop_current_action()

            if self.socket_sub:
                self.socket_sub.close()
            if self.context:
                self.context.term()

            try:
                self.robot_state_client.ServiceSwitch("sport_mode", 1)
                self.sport_client.StandDown()
            except:
                pass

def signal_handler(signum, frame):
    """Signal handler."""
    print("\n\nExit signal received...")
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Go2 Monkey ZMQ Controller")
    print("Remote control via ZMQ")
    print("\n⚠️ Robot will move!")
    print("\nAuto start in 3 seconds...")
    time.sleep(3)

    controller = Go2MonkeyZMQController()
    controller.run()

    print("\nProgram ended")
    sys.exit(0)
import argparse
import yaml

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--score", type=float, required=True)
    p.add_argument("--playbook", type=str, default="examples/sample_playbook.yaml")
    return p.parse_args()

def restart_container(target: str):
    print(f"[PLAYBOOK] restart_container: {target}")
    # 실제 환경에서는 docker/k8s 명령 호출로 교체하세요.

def remove_from_load_balancer(target: str):
    print(f"[PLAYBOOK] remove_from_load_balancer: {target}")
    # 실제 환경에서는 LB 설정 변경 API 호출로 교체하세요.

def send_alert(channel: str, message: str):
    print(f"[PLAYBOOK] send_alert via {channel}: {message}")
    # 실제 환경에서는 Telegram/Slack webhook 호출로 교체하세요.

def execute_playbook(playbook):
    for action in playbook.get("actions", []):
        t = action.get("type")
        if t == "restart_container":
            restart_container(action.get("target", "unknown"))
        elif t == "remove_from_load_balancer":
            remove_from_load_balancer(action.get("target", "unknown"))
        elif t == "send_alert":
            send_alert(action.get("channel", "unknown"), action.get("message", "no message"))
        else:
            print(f"[PLAYBOOK] unknown action type: {t}")

def main():
    args = parse_args()
    score = args.score
    print(f"[RESPONDER] score={score:.4f}, threshold={args.threshold:.4f}")
    if score >= args.threshold:
        print("[RESPONDER] score >= threshold, executing playbook...")
        with open(args.playbook, "r", encoding="utf-8") as f:
            playbook = yaml.safe_load(f)
        execute_playbook(playbook)
    else:
        print("[RESPONDER] score < threshold, no action.")

if __name__ == "__main__":
    main()

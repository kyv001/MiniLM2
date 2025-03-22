from matplotlib import pyplot as plt

def load_log(lines: list[str]) -> tuple[list[float], list[float], list[float], list[int], list[int], list[float], list[float]]:

    losses: list[float] = []
    timestamps: list[float] = []
    val_losses: list[float] = []
    val_timestamps: list[float] = []
    lrs: list[float] = []
    steps: list[int] = []
    val_steps: list[int] = []
    t0: float | None = None
    for line in lines:
        loss_type, step, lr, loss, time = line.strip().split(",")
        if t0 is None:
            t0 = float(time)
        if loss_type in ("TRAIN", "SFT"):
            losses.append(float(loss))
            lrs.append(float(lr))
            steps.append(int(step))
            timestamps.append(float(time) - t0)
        elif loss_type == "VAL":
            val_losses.append(float(loss))
            val_steps.append(int(step))
            val_timestamps.append(timestamps[-1])
            t0 += float(time) - t0 - timestamps[-1]
    return losses, val_losses, lrs, steps, val_steps, timestamps, val_timestamps

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m minilm2.utils.show_log <log_file>")
        exit(1)
    log_file = sys.argv[1]
    with open(log_file, 'r') as f:
        lines = f.readlines()
    losses, val_losses, lrs, steps, val_steps, timestamps, val_timestamps = load_log(lines)
    plt.plot(steps, losses, label="train loss")
    plt.plot(val_steps, val_losses, label="val loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    plt.plot(timestamps, losses, label="train loss")
    plt.plot(val_timestamps, val_losses, label="val loss")
    plt.xlabel("time/s")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    
    plt.plot(steps, lrs, label="lr")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

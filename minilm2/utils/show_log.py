from matplotlib import pyplot as plt

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m minilm2.utils.show_log <log_file>")
        exit(1)
    log_file = sys.argv[1]
    with open(log_file, 'r') as f:
        lines = f.readlines()
    losses: list[float] = []
    val_losses: list[float] = []
    lrs: list[float] = []
    steps: list[int] = []
    val_steps: list[int] = []
    for line in lines:
        loss_type, step, lr, loss = line.strip().split(",")
        if loss_type in ("TRAIN", "SFT"):
            losses.append(float(loss))
            lrs.append(float(lr))
            steps.append(int(step))
        elif loss_type == "VAL":
            val_losses.append(float(loss))
            val_steps.append(int(step))
    plt.plot(steps, losses, label="train loss")
    plt.plot(val_steps, val_losses, label="val loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    
    plt.plot(steps, lrs, label="lr")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

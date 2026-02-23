# KeepData.py (rev: show per-finger names in HUD/panel/print-every)
# rev: add real-time graph with --graph, divide values by 100, fix legend display
# Collects data from ESP32 (Serial CSV: t_ms + 5 values) into a single file (multi-label)
# Data mode: Automatically detects whether "raw (int)" or "ohm (float/inf)"
# Requires: pip install pyserial
# For graphing: pip install matplotlib
# Example:
# python KeepData.py --port /dev/cu.usbserial-0001 --labels fist,open,pinch --ema 0.25 --hud --print-every 50 --panel --graph
import argparse, csv, os, sys, time, threading, re, math
from datetime import datetime
import serial
from serial.tools import list_ports

FINGER_KEYS = ["Thumb", "Index", "Middle", "Ring", "Little"]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_labels(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def show_help(preset):
    print("\nCommands:")
    print(" start / stop / toggle")
    print(" label <name> Set current label")
    if preset:
        print(" 1..9 Select label from preset:", ", ".join(preset[:9]))
    print(" list Show current label + statistics")
    print(" hud Toggle HUD (on/off)")
    print(" panel Toggle status panel before commands")
    print(" clear | cls Clear screen")
    print(" help Show commands")
    print(" quit Exit and save\n")


def print_quick_commands(preset):
    base = "Commands: start | stop | toggle | label <name> | list | hud | panel | clear | quit"
    if preset:
        base += " | 1..9 Select preset"
    print(base)


def try_float(s):
    s = s.strip()
    if re.fullmatch(r"(?i)[+-]?(inf(inity)?|nan)", s):
        if s.lower().startswith("nan"):
            return float("nan")
        return float("inf") if not s.startswith("-") else float("-inf")
    return float(s)


def parse_line(line):
    line = line.strip()
    if not line:
        return None
    if line.lower().startswith("t_ms"):
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 6:
        return None
    t_ms = parts[0]
    vals = []
    for s in parts[1:6]:
        try:
            vals.append(try_float(s))
        except Exception:
            return None
    return t_ms, vals


def detect_mode(vals):
    for v in vals:
        if math.isinf(v) or math.isnan(v):
            return "ohm"
        if abs(v - int(v)) > 1e-9:
            return "ohm"
    return "raw"


def format_vals_named(vals, mode, names=FINGER_NAMES, max_items=5, prec=1):
    """Returns string: 'Thumb=12.3Ω, Index=...' or for raw: 'Thumb=10.23, ...'"""
    items = []
    unit = "Ω" if mode == "ohm" else ""
    for i, v in enumerate(vals[:max_items]):
        name = names[i] if i < len(names) else f"ch{i}"
        if v is None:
            items.append(f"{name}=—")
        elif math.isinf(v):
            items.append(f"{name}=inf{unit}")
        elif math.isnan(v):
            items.append(f"{name}=nan{unit}")
        elif mode == "ohm":
            items.append(f"{name}={v:.{prec}f}{unit}")
        else:
            items.append(f"{name}={v:.{prec}f}")
    return ", ".join(items)


def draw_panel(get_state, width=86):
    top = "┌" + "─" * (width - 2) + "┐"
    bot = "└" + "─" * (width - 2) + "┘"

    def line(s):
        s = s[: width - 4]
        return "│ " + s + " " * (width - 4 - len(s)) + " │"

    st = get_state()
    rows = [
        top,
        line(
            f"REC={'ON' if st['recording'] else 'OFF'} | label={st['label']} | rows={st['row_count']} | rate={st['rps']:.1f} rows/s"
            if st["rps"] is not None
            else f"REC={'ON' if st['recording'] else 'OFF'} | label={st['label']} | rows={st['row_count']}"
        ),
        line(f"t={st['t_ms']}" if st["t_ms"] is not None else "t=—"),
    ]
    vals_line = (
        "val: " + format_vals_named(st["vals"], st["mode"])
        if st["vals"][0] is not None
        else "val=—"
    )
    rows.append(line(vals_line))
    if st["use_ema"]:
        ema_line = (
            "ema: " + format_vals_named(st["ema"], st["mode"])
            if st["ema"][0] is not None
            else "ema=—"
        )
        rows.append(line(ema_line))
    rows.append(bot)
    print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--port", required=False, help="e.g., COM5 or /dev/cu.usbserial-xxxx"
    )
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--labels", default="", help="preset labels, e.g., fist,open,pinch")
    ap.add_argument(
        "--out",
        default=None,
        help="Custom output file (auto-generated if not specified)",
    )
    ap.add_argument("--outdir", default="data", help="Output directory")
    ap.add_argument(
        "--ema", type=float, default=0.0, help="EMA alpha (0=off, e.g., 0.25)"
    )
    ap.add_argument(
        "--list-ports", action="store_true", help="List available ports and exit"
    )
    ap.add_argument("--hud", action="store_true", help="Show live HUD on start")
    ap.add_argument(
        "--hud-interval", type=float, default=0.5, help="Seconds per HUD update"
    )
    ap.add_argument(
        "--print-every",
        type=int,
        default=0,
        help="Print sample line every N rows (0=off)",
    )
    ap.add_argument(
        "--panel", action="store_true", help="Show status panel before each command"
    )
    ap.add_argument("--panel-width", type=int, default=86, help="Status panel width")
    ap.add_argument(
        "--graph", action="store_true", help="Show real-time resistance and EMA graph"
    )
    ap.add_argument(
        "--graph-history", type=float, default=60.0, help="Seconds of history in graph"
    )
    ap.add_argument(
        "--graph-interval", type=float, default=0.2, help="Seconds per graph update"
    )
    args = ap.parse_args()

    if args.graph:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation

            plt.ion()  # Interactive mode for non-blocking
        except ImportError:
            print("matplotlib is required for --graph: pip install matplotlib")
            sys.exit(1)

    if args.list_ports:
        print("Serial ports:")
        for p in list_ports.comports():
            print(f" - {p.device} ({p.description})")
        return

    if not args.port:
        ports = list(list_ports.comports())
        if ports:
            args.port = ports[0].device
            print(f"* Auto-picked port: {args.port}")
    if not args.port:
        print("Please specify --port or use --list-ports to check available ports")
        sys.exit(1)

    preset = parse_labels(args.labels)
    os.makedirs(args.outdir, exist_ok=True)
    outpath = args.out or os.path.join(args.outdir, f"session_{timestamp()}.csv")
    print(f"Opening Serial {args.port} @ {args.baud} ...")
    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(0.6)
    f = open(outpath, "w", newline="", encoding="utf-8")
    w = csv.writer(f)

    # Shared state
    current_label = "none"
    recording = False
    alive = True
    row_count = 0
    per_label = {}
    current_ema = [None] * 5
    FLUSH_EVERY = 200
    lock = threading.Lock()
    last_t_ms = None
    last_vals = [None] * 5
    last_ema = [None] * 5
    hud_on = args.hud
    hud_pause = False
    last_rate_rows = 0
    last_rate_time = time.time()
    smoothed_rps = None  # rows per second (EMA)
    use_ema = args.ema > 0
    mode = None  # "raw" or "ohm"
    hist_t = []  # in seconds
    hist_vals = [[] for _ in range(5)]
    hist_ema = [[] for _ in range(5)]
    history_sec = args.graph_history

    def headers_for_mode(m):
        base = ["t_ms"]
        prefix = "raw" if m == "raw" else "ohm"
        base += [f"{prefix}_{k}" for k in FINGER_KEYS]
        if use_ema:
            base += [f"ema_{k}" for k in FINGER_KEYS]
        base += ["label"]
        return base

    def get_state():
        with lock:
            return {
                "recording": recording,
                "label": current_label,
                "row_count": row_count,
                "rps": smoothed_rps,
                "t_ms": last_t_ms,
                "vals": last_vals[:],
                "ema": last_ema[:],
                "use_ema": use_ema,
                "mode": mode if mode else "raw",
            }

    def reader():
        nonlocal row_count, last_t_ms, last_vals, last_ema, smoothed_rps, last_rate_rows, last_rate_time, mode
        header_written = False
        while alive:
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                parsed = parse_line(line)
                if not parsed:
                    continue
                t_ms, vals = parsed
                # Divide values by 1000
                vals = [
                    v / 1000 if not (math.isinf(v) or math.isnan(v)) else v
                    for v in vals
                ]
                t_sec = float(t_ms) / 1000.0 if t_ms.isdigit() else 0.0
                if mode is None:
                    mode = detect_mode(vals)
                    w.writerow(headers_for_mode(mode))
                    f.flush()
                    header_written = True
                    print(
                        f"* Detected mode: {mode} ({'float/Ω' if mode=='ohm' else 'int/raw'})"
                    )
                with lock:
                    last_t_ms = t_ms
                    last_vals = vals
                    # Prepare values for history (replace inf with nan for plotting)
                    plot_vals = [float("nan") if math.isinf(v) else v for v in vals]
                    hist_t.append(t_sec)
                    for i in range(5):
                        hist_vals[i].append(plot_vals[i])
                    if use_ema:
                        for i in range(5):
                            if current_ema[i] is None or math.isnan(current_ema[i]):
                                current_ema[i] = vals[i]
                            else:
                                current_ema[i] = (
                                    args.ema * vals[i]
                                    + (1.0 - args.ema) * current_ema[i]
                                )
                        last_ema = [
                            float("nan") if math.isinf(e) else e for e in current_ema
                        ]
                        for i in range(5):
                            hist_ema[i].append(last_ema[i])
                    else:
                        last_ema = [None] * 5
                    # Trim history
                    while hist_t and hist_t[0] < hist_t[-1] - history_sec:
                        hist_t.pop(0)
                        for i in range(5):
                            hist_vals[i].pop(0)
                            if use_ema:
                                hist_ema[i].pop(0)
                if recording:
                    row_out = [t_ms] + vals
                    if use_ema:
                        row_out += [
                            (float("nan") if current_ema[i] is None else current_ema[i])
                            for i in range(5)
                        ]
                    row_out += [current_label]
                    w.writerow(row_out)
                    row_count += 1
                    per_label[current_label] = per_label.get(current_label, 0) + 1
                    if args.print_every and (row_count % args.print_every == 0):
                        named_vals = format_vals_named(vals, mode, FINGER_NAMES, prec=1)
                        pr_ema = ""
                        if use_ema:
                            named_ema = format_vals_named(
                                current_ema, mode, FINGER_NAMES, prec=1
                            )
                            pr_ema = f" | ema: {named_ema}"
                        print(
                            f"[line {row_count}] label={current_label} t={t_ms} | {named_vals}{pr_ema}"
                        )
                    if row_count % FLUSH_EVERY == 0:
                        f.flush()
                    now = time.time()
                    dt = now - last_rate_time
                    if dt >= 0.5:
                        rps = (row_count - last_rate_rows) / dt
                        last_rate_rows = row_count
                        last_rate_time = now
                        smoothed_rps = (
                            rps
                            if smoothed_rps is None
                            else 0.3 * rps + 0.7 * smoothed_rps
                        )
            except Exception:
                time.sleep(0.02)

    def hud():
        while alive:
            if hud_on and not hud_pause:
                st = get_state()
                parts = []
                parts.append(f"REC={'ON' if st['recording'] else 'OFF'}")
                parts.append(f"label={st['label']}")
                if st["rps"] is not None:
                    parts.append(f"rate={st['rps']:.1f} rows/s")
                parts.append(f"rows={st['row_count']}")
                if st["t_ms"] is not None:
                    parts.append(f"t={st['t_ms']}")
                if st["vals"][0] is not None:
                    parts.append(
                        "val: "
                        + format_vals_named(
                            st["vals"], st["mode"], FINGER_NAMES, prec=1
                        )
                    )
                if st["use_ema"] and st["ema"][0] is not None:
                    parts.append(
                        "ema: "
                        + format_vals_named(st["ema"], st["mode"], FINGER_NAMES, prec=1)
                    )
                line = " | ".join(parts)
                print("\r" + line + " " * 20, end="", flush=True)
            time.sleep(max(0.1, args.hud_interval))

    t_reader = threading.Thread(target=reader, daemon=True)
    t_reader.start()
    t_hud = threading.Thread(target=hud, daemon=True)
    t_hud.start()

    if args.graph:
        fig, axs = plt.subplots(2 if use_ema else 1, 1, sharex=True, figsize=(10, 6))
        if not use_ema:
            axs = [axs]
        lines_vals = [axs[0].plot([], [], label=FINGER_NAMES[i])[0] for i in range(5)]
        if use_ema:
            lines_ema = [
                axs[1].plot([], [], label=FINGER_NAMES[i])[0] for i in range(5)
            ]
        # Improved legend placement
        axs[0].legend(
            loc="upper right", bbox_to_anchor=(1.15, 1), fontsize=8, framealpha=0.8
        )
        if use_ema:
            axs[1].legend(
                loc="upper right", bbox_to_anchor=(1.15, 1), fontsize=8, framealpha=0.8
            )
        axs[0].set_title("Values")
        axs[0].set_ylabel("Value")
        if use_ema:
            axs[1].set_title("EMA")
            axs[1].set_ylabel("Value")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        first_update = [True]  # Mutable for nonlocal

        def init():
            for ax in axs:
                ax.set_xlim(0, history_sec)
            return lines_vals + (lines_ema if use_ema else [])

        def update(frame):
            with lock:
                t = hist_t[:]
                v = [h[:] for h in hist_vals]
                e = [h[:] for h in hist_ema] if use_ema else None
            if not t:
                return init()
            min_t = max(0, t[-1] - history_sec)
            idx = next((i for i, x in enumerate(t) if x >= min_t), 0)
            t = t[idx:]
            for i in range(5):
                lines_vals[i].set_data(t, v[i][idx:])
            if use_ema:
                for i in range(5):
                    lines_ema[i].set_data(t, e[i][idx:])
            for ax in axs:
                ax.set_xlim(min_t, t[-1] + 1)
                ax.relim()
                ax.autoscale_view(True, True, True)
            if first_update[0] and mode:
                unit = " (Ω)" if mode == "ohm" else " (raw)"
                axs[0].set_title("Resistance" + unit)
                axs[0].set_ylabel("Value" + unit)
                if use_ema:
                    axs[1].set_title("EMA" + unit)
                    axs[1].set_ylabel("Value" + unit)
                first_update[0] = False
            return lines_vals + (lines_ema if use_ema else [])

        ani = FuncAnimation(
            fig, update, init_func=init, blit=False, interval=args.graph_interval * 1000
        )
        plt.show(block=False)

    print(f"Saving to file: {outpath}")
    print(f"EMA: {'ON (alpha='+str(args.ema)+')' if use_ema else 'OFF'}")
    if preset:
        print(
            "Preset:", ", ".join(f"{i+1}:{name}" for i, name in enumerate(preset[:9]))
        )
    print("Not recording yet — type 'start' to begin (type 'help' for commands)")
    show_help(preset)

    try:
        while True:
            hud_pause = True
            if args.panel:
                print("\r" + " " * 160 + "\r", end="", flush=True)
                draw_panel(get_state, width=args.panel_width)
            print_quick_commands(preset)
            if args.graph:
                plt.pause(0.001)  # Process matplotlib events
            cmd = input("> ").strip()
            low = cmd.lower()
            if low in ("quit", "exit"):
                break
            elif low == "help":
                show_help(preset)
            elif low == "start":
                recording = True
                print("** START **")
            elif low == "stop":
                recording = False
                print("** STOP **")
            elif low == "toggle":
                recording = not recording
                print(f"** {'START' if recording else 'STOP'} **")
            elif low.startswith("label "):
                current_label = cmd.split(" ", 1)[1].strip() or "none"
                print(f"label = {current_label}")
            elif low == "list":
                st = get_state()
                print(f"Current label: {st['label']}")
                if preset:
                    print(
                        "Preset:",
                        ", ".join(f"{i+1}:{name}" for i, name in enumerate(preset[:9])),
                    )
                if per_label:
                    print(
                        "Count per label:",
                        ", ".join(f"{k}:{v}" for k, v in per_label.items()),
                    )
            elif low == "hud":
                hud_on = not hud_on
                print(f"HUD = {'ON' if hud_on else 'OFF'}")
            elif low == "panel":
                args.panel = not args.panel
                print(f"Panel = {'ON' if args.panel else 'OFF'}")
            elif low in ("clear", "cls"):
                os.system("clear")
                if args.panel:
                    print("\r" + " " * 160 + "\r", end="", flush=True)
                    draw_panel(get_state, width=args.panel_width)
                print_quick_commands(preset)
            elif preset and low.isdigit() and 1 <= int(low) <= min(9, len(preset)):
                current_label = preset[int(low) - 1]
                print(f"label = {current_label} (preset #{low})")
            else:
                print("Unknown command (type 'help')")
            hud_pause = False
    except KeyboardInterrupt:
        pass
    finally:
        alive = False
        time.sleep(0.2)
        try:
            ser.close()
        except:
            pass
        try:
            f.flush()
            f.close()
        except:
            pass
        print("\r" + " " * 200 + "\r", end="")
        print("\nSummary:")
        print(f" File: {outpath}")
        print(f" Total rows: {row_count}")
        if per_label:
            print(" Per label:", ", ".join(f"{k}:{v}" for k, v in per_label.items()))
        print("bye.")


if __name__ == "__main__":
    main()

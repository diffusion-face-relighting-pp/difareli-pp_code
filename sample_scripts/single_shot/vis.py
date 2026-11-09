import argparse
import glob
import json
import os
import re
import numpy as np
from flask import Flask, request, send_from_directory


def parse_args():
    parser = argparse.ArgumentParser(description="Flask visualization server for DiFaReli++ relit_output.")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/data/mint/DiFaReli++/examples/relit_output",
        help="Root directory containing relighting/reshadowing results.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/data/mint/DiFaReli++/examples/relighting_data",
        help="Fallback dataset root for input images.",
    )
    parser.add_argument("--set", dest="set_", type=str, default="valid", help="Fallback dataset split.")
    parser.add_argument("--ds", type=int, default=10, help="Max frames to display per row (0 = show all).")
    parser.add_argument("--port", type=int, default=8888, help="Port to run Flask server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind Flask server to.")
    return parser.parse_args()


def sort_by_frame(path_list):
    def extract_idx(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"frame(\d+)$", stem)
        return int(m.group(1)) if m else 0

    return sorted(path_list, key=extract_idx)


def subsample_frames(frames, ds):
    if ds <= 0 or len(frames) <= ds:
        return frames
    indices = np.linspace(0, len(frames) - 1, num=ds, dtype=int)
    seen = set()
    selected = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            selected.append(frames[idx])
    return selected


def parse_run_dir(run_dir, root_dir, default_dataset_root, default_set):
    """
    Parses a single subject result directory (either new `<model>/<subj>/{frames,vid,res_desc.json}`
    layout from relight_fixed.py or legacy `.../src=<src>/dst=<dst>/Lerp_*/n_frames=*/` layout).
    """
    frames_dir = os.path.join(run_dir, "frames")
    vid_subdir = os.path.join(run_dir, "vid")
    is_new_layout = os.path.isdir(frames_dir) or os.path.isdir(vid_subdir)
    if not os.path.isdir(frames_dir):
        frames_dir = run_dir

    vid_dir = vid_subdir if os.path.isdir(vid_subdir) else run_dir

    res_frames = sort_by_frame(glob.glob(os.path.join(frames_dir, "res_frame*.png")))
    all_mp4s = sorted(glob.glob(os.path.join(vid_dir, "*.mp4"))) or sorted(glob.glob(os.path.join(run_dir, "*.mp4")))
    if not res_frames and not all_mp4s:
        return None

    desc = {}
    desc_path = os.path.join(run_dir, "res_desc.json")
    if os.path.isfile(desc_path):
        try:
            with open(desc_path, "r") as f:
                desc = json.load(f)
        except Exception:
            desc = {}

    samples_meta = desc.get("samples", {})
    sampling_args = desc.get("sampling_args", {})
    dataset_meta = desc.get("dataset", {})

    src_id = samples_meta.get("src_id")
    dst_id = samples_meta.get("dst_id")

    rel_run = os.path.relpath(run_dir, root_dir)

    if is_new_layout:
        subj_key = os.path.basename(run_dir)
        model_name = os.path.relpath(os.path.dirname(run_dir), root_dir)
        if model_name == ".":
            model_name = os.path.basename(os.path.abspath(root_dir))
    else:
        # Legacy path e.g. relight/cfg=.../valid/render_face/src=max_1.png/dst=cr7_1.png/Lerp_1000/n_frames=60
        parts = rel_run.split(os.sep)
        src_part_idx = next((i for i, p in enumerate(parts) if p.startswith("src=")), None)
        if src_part_idx is not None:
            if not src_id:
                src_id = parts[src_part_idx][len("src="):]
            if src_part_idx + 1 < len(parts) and parts[src_part_idx + 1].startswith("dst=") and not dst_id:
                dst_id = parts[src_part_idx + 1][len("dst="):]
            prefix = os.sep.join(parts[:src_part_idx])
            suffix = os.sep.join(parts[src_part_idx + 2:])
            model_name = f"{prefix} ({suffix})" if suffix else prefix
            subj_key = os.path.splitext(src_id)[0] if src_id else parts[src_part_idx]
        else:
            subj_key = os.path.basename(run_dir)
            model_name = os.path.dirname(rel_run)

    if not src_id:
        src_id = f"{subj_key}.png"
    if not dst_id:
        dst_id = src_id

    ds_root = (
        dataset_meta.get("dataset_root")
        or sampling_args.get("dataset_root")
        or default_dataset_root
    )
    ds_set = dataset_meta.get("set") or sampling_args.get("set") or default_set
    img_dir = os.path.join(ds_root, "images_aligned", ds_set) if ds_root else None
    src_img_path = os.path.join(img_dir, src_id) if img_dir else None
    dst_img_path = os.path.join(img_dir, dst_id) if img_dir else None

    src_ren_frames = sort_by_frame(glob.glob(os.path.join(frames_dir, "src_ren_frame*.png")))
    dst_ren_frames = sort_by_frame(glob.glob(os.path.join(frames_dir, "dst_ren_frame*.png")))
    src_shad_frames = sort_by_frame(glob.glob(os.path.join(frames_dir, "src_shadm_shad_frame*.png")))
    dst_shad_frames = sort_by_frame(glob.glob(os.path.join(frames_dir, "dst_shadm_shad_frame*.png")))

    res_vid = None
    for search_vdir in (vid_dir, run_dir):
        for vname in ("res_rt.mp4", "res.mp4", "out_rt.mp4", "out.mp4"):
            vpath = os.path.join(search_vdir, vname)
            if os.path.isfile(vpath):
                res_vid = vpath
                break
        if res_vid:
            break
    if not res_vid and all_mp4s:
        res_vid = all_mp4s[0]

    return {
        "subj_key": subj_key,
        "model_name": model_name,
        "run_dir": run_dir,
        "src_id": src_id,
        "dst_id": dst_id,
        "src_img_path": src_img_path if (src_img_path and os.path.isfile(src_img_path)) else None,
        "dst_img_path": dst_img_path if (dst_img_path and os.path.isfile(dst_img_path)) else None,
        "res_frames": res_frames,
        "src_ren_frames": src_ren_frames,
        "dst_ren_frames": dst_ren_frames,
        "src_shad_frames": src_shad_frames,
        "dst_shad_frames": dst_shad_frames,
        "res_vid": res_vid,
    }


def discover_runs(root_dir, default_dataset_root, default_set):
    """Walks root_dir to find all subject run directories."""
    candidate_dirs = set()
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "res_desc.json" in filenames:
            candidate_dirs.add(dirpath)
        elif any(f.startswith("res_frame") and f.endswith(".png") for f in filenames):
            if os.path.basename(dirpath) == "frames":
                candidate_dirs.add(os.path.dirname(dirpath))
            else:
                candidate_dirs.add(dirpath)
        elif any(f.endswith(".mp4") for f in filenames):
            if os.path.basename(dirpath) == "vid":
                candidate_dirs.add(os.path.dirname(dirpath))
            else:
                candidate_dirs.add(dirpath)

    runs = []
    for d in sorted(candidate_dirs):
        parsed = parse_run_dir(d, root_dir, default_dataset_root, default_set)
        if parsed is not None:
            runs.append(parsed)
    return runs


def render_frames_html(frames, ds, img_size=128):
    if not frames:
        return "-"
    selected = subsample_frames(frames, ds)
    imgs = [
        f'<img width="{img_size}" height="{img_size}" src="/files/{f}" title="{os.path.basename(f)}">'
        for f in selected
    ]
    return "".join(imgs)


def create_app(args):
    app = Flask(__name__)

    @app.route("/files/<path:path>")
    def servefile(path):
        return send_from_directory("/", path)

    @app.route("/")
    def root():
        root_dir = os.path.abspath(request.args.get("dir", args.root_dir))
        dataset_root = os.path.abspath(args.dataset_root) if args.dataset_root else None
        sub_filter = request.args.get("sub", "")
        group_by = request.args.get("group_by", "subject")  # 'subject' or 'model'
        ds = int(request.args.get("ds", args.ds))
        img_size = int(request.args.get("size", 128))
        show_vid = request.args.get("show_vid", "True") == "True"
        show_ren = request.args.get("show_ren", "False") == "True"
        show_shadm = request.args.get("show_shadm", "False") == "True"

        search_dir = os.path.join(root_dir, sub_filter) if sub_filter else root_dir
        all_runs = discover_runs(root_dir, dataset_root, args.set_)
        runs = discover_runs(search_dir, dataset_root, args.set_) if sub_filter else all_runs

        out = """<html><head><title>DiFaReli++ Visualization</title>
        <style>
            body { background-color: white; color: black; font-family: sans-serif; }
            table { border-collapse: collapse; margin-bottom: 20px; }
            th, tr, td { border: 1px solid black; padding: 4px; text-align: center; vertical-align: middle; white-space: nowrap; }
        </style>
        <script>
            function setParam(key, value) {
                var url = new URL(window.location.href);
                url.searchParams.set(key, value);
                window.location.href = url.toString();
            }
        </script>
        </head><body>"""

        out += f"<h3>Directory: {search_dir}</h3>"

        # Toggle buttons for display options
        next_group = "model" if group_by == "subject" else "subject"
        next_vid = "False" if show_vid else "True"
        next_ren = "False" if show_ren else "True"
        next_shadm = "False" if show_shadm else "True"

        out += "<p>"
        out += f"<button onclick=\"setParam('group_by', '{next_group}')\">Group By: {group_by} (switch to {next_group})</button> "
        out += f"<button onclick=\"setParam('show_vid', '{next_vid}')\">show_vid: {show_vid}</button> "
        out += f"<button onclick=\"setParam('show_ren', '{next_ren}')\">show_ren: {show_ren}</button> "
        out += f"<button onclick=\"setParam('show_shadm', '{next_shadm}')\">show_shadm: {show_shadm}</button> "
        out += "&nbsp;|&nbsp; ds (frames): "
        for ds_val in (2, 5, 10, 0):
            label_ds = "All (0)" if ds_val == 0 else str(ds_val)
            active = "*" if ds == ds_val else ""
            out += f"<button onclick=\"setParam('ds', '{ds_val}')\">{label_ds}{active}</button> "
        out += "&nbsp;|&nbsp; size: "
        for sz_val in (128, 192, 256):
            active = "*" if img_size == sz_val else ""
            out += f"<button onclick=\"setParam('size', '{sz_val}')\">{sz_val}{active}</button> "
        out += "</p>"

        # Subfolder filter buttons
        all_models_root = sorted({r["model_name"].split(" (")[0] for r in all_runs})
        top_subs = sorted({m.split(os.sep)[0] for m in all_models_root if m})
        sub_candidates = [""] + top_subs + [m for m in all_models_root if m not in top_subs]
        out += "<p>Subfolder: "
        for sc in sub_candidates:
            lbl = "All" if sc == "" else sc
            active = " *" if sub_filter == sc else ""
            out += f"<button onclick=\"setParam('sub', '{sc}')\">{lbl}{active}</button> "
        out += "</p>"

        all_models = sorted({r["model_name"] for r in runs})
        all_subjects = sorted({r["subj_key"] for r in runs})
        out += (
            f"<p>Subjects ({len(all_subjects)}): {', '.join(all_subjects)}<br>"
            f"Subfolders / Models ({len(all_models)}): {', '.join(all_models)}</p><hr>"
        )

        if not runs:
            out += "<p>No results found.</p></body></html>"
            return out

        if group_by == "model":
            # Group by model/subfolder: one table per model, each row is a subject
            by_model = {}
            for r in runs:
                by_model.setdefault(r["model_name"], []).append(r)

            for model_name in sorted(by_model.keys()):
                model_runs = sorted(by_model[model_name], key=lambda x: x["subj_key"])
                out += f"<h4>Model / Subfolder: {model_name}</h4>"
                out += "<table>"
                out += "<tr><th>Subject</th><th>Input (Src / Dst)</th>"
                if show_vid:
                    out += "<th>Video</th>"
                out += "<th>Relit Frames (res)</th>"
                if show_ren:
                    out += "<th>Render (src_ren / dst_ren)</th>"
                if show_shadm:
                    out += "<th>Shadow Mask (src_shadm / dst_shadm)</th>"
                out += "</tr>"

                for r in model_runs:
                    out += render_row_html(
                        label=f"{r['subj_key']}<br><small>({r['src_id']} &rarr; {r['dst_id']})</small>",
                        r=r,
                        ds=ds,
                        img_size=img_size,
                        show_vid=show_vid,
                        show_ren=show_ren,
                        show_shadm=show_shadm,
                    )
                out += "</table><hr>"
        else:
            # Default: Group by subject: one table per subject, each row is a model/subfolder for comparison
            by_subj = {}
            for r in runs:
                by_subj.setdefault(r["subj_key"], []).append(r)

            for subj_key in sorted(by_subj.keys()):
                subj_runs = sorted(by_subj[subj_key], key=lambda x: x["model_name"])
                r0 = subj_runs[0]
                out += f"<h4>Subject: {subj_key} ({r0['src_id']} &rarr; {r0['dst_id']})</h4>"
                out += "<table>"
                out += "<tr><th>Model / Subfolder</th><th>Input (Src / Dst)</th>"
                if show_vid:
                    out += "<th>Video</th>"
                out += "<th>Relit Frames (res)</th>"
                if show_ren:
                    out += "<th>Render (src_ren / dst_ren)</th>"
                if show_shadm:
                    out += "<th>Shadow Mask (src_shadm / dst_shadm)</th>"
                out += "</tr>"

                for r in subj_runs:
                    out += render_row_html(
                        label=r["model_name"],
                        r=r,
                        ds=ds,
                        img_size=img_size,
                        show_vid=show_vid,
                        show_ren=show_ren,
                        show_shadm=show_shadm,
                    )
                out += "</table><hr>"

        out += "</body></html>"
        return out

    return app


def render_row_html(label, r, ds, img_size, show_vid, show_ren, show_shadm):
    row = "<tr>"
    row += f"<td>{label}</td>"

    # Input (Src / Dst)
    row += "<td>"
    if r["src_img_path"]:
        row += f'<img width="{img_size}" height="{img_size}" src="/files/{r["src_img_path"]}" title="src: {r["src_id"]}">'
    elif r["res_frames"]:
        row += f'<img width="{img_size}" height="{img_size}" src="/files/{r["res_frames"][0]}" title="frame0">'
    if r["dst_img_path"] and r["dst_id"] != r["src_id"]:
        row += f'<img width="{img_size}" height="{img_size}" src="/files/{r["dst_img_path"]}" title="dst: {r["dst_id"]}">'
    row += "</td>"

    # Video
    if show_vid:
        row += "<td>"
        if r["res_vid"]:
            row += (
                f'<video width="{img_size}" height="{img_size}" controls autoplay muted loop>'
                f'<source src="/files/{r["res_vid"]}" type="video/mp4">'
                f"</video>"
            )
        else:
            row += "-"
        row += "</td>"

    # Relit frames
    row += f"<td>{render_frames_html(r['res_frames'], ds, img_size)}</td>"

    # Render frames
    if show_ren:
        row += "<td>"
        row += render_frames_html(r["src_ren_frames"], ds, img_size)
        if r["dst_ren_frames"]:
            row += "<br>" + render_frames_html(r["dst_ren_frames"], ds, img_size)
        row += "</td>"

    # Shadow mask frames
    if show_shadm:
        row += "<td>"
        row += render_frames_html(r["src_shad_frames"], ds, img_size)
        if r["dst_shad_frames"]:
            row += "<br>" + render_frames_html(r["dst_shad_frames"], ds, img_size)
        row += "</td>"

    row += "</tr>"
    return row


if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port, debug=True, threaded=False)

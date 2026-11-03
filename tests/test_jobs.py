import time

from PIL import Image

from enhancer import imageio, jobs
from enhancer.pipeline import Settings
from enhancer.tiling import Cancelled
from tests.conftest import photo_like


def _photo(path):
    Image.fromarray(imageio.to_uint8(photo_like(24, 32))).save(path)
    return path


def test_queue_order_failure_isolation_and_cancel(tmp_path, monkeypatch):
    ran = []

    def fake_process(img, s, rt, progress=None, cancel=None, report=None):
        name = ran_name[0]
        ran.append(name)
        if name == "bad.png":
            raise RuntimeError("boom")
        if name == "slow.png":
            for _ in range(200):
                if cancel():
                    raise Cancelled()
                time.sleep(0.01)
        progress(1.0, "Saving")
        return img.repeat(2, 0).repeat(2, 1)

    ran_name = [""]
    real_load = imageio.load

    def spy_load(p):
        ran_name[0] = p.name
        return real_load(p)

    monkeypatch.setattr(jobs, "process", fake_process)
    monkeypatch.setattr(jobs.imageio, "load", spy_load)
    files = [_photo(tmp_path / n) for n in ("a.png", "bad.png", "slow.png", "b.png")]
    (tmp_path / "notes.txt").write_text("x")
    q = jobs.JobQueue(rt=type("RT", (), {"notice": ""})(), preview_dir=tmp_path / "prev")
    added = q.add(files + [tmp_path / "notes.txt"], Settings(upscale=2), tmp_path / "out")
    assert len(added) == 4  # the .txt is skipped
    while q.get(added[2].id).status != "running":
        time.sleep(0.01)
    q.cancel(added[2].id)
    assert q.wait_idle(10)
    st = {j.src.name: j.status for j in q.list()}
    assert st == {"a.png": "done", "bad.png": "error", "slow.png": "cancelled", "b.png": "done"}
    assert ran == ["a.png", "bad.png", "slow.png", "b.png"]
    done = q.get(added[0].id)
    assert done.out.exists() and done.out_size == (64, 48) and done.after.exists()
    assert "boom" in q.get(added[1].id).error
    q.shutdown()

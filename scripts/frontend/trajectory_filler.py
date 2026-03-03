import torch
import lietorch
from lietorch import SE3


class PoseTrajectoryFiller:
    """Fill in camera poses for non-keyframe images via SE3 interpolation.

    Uses the keyframe timestamps and poses stored in the DepthVideo buffer to
    linearly interpolate (in the Lie algebra) a pose for every input frame.
    No GPU memory beyond the video buffer is required.
    """

    def __init__(self, net, video, cfg, device="cuda:0"):
        self.video  = video
        self.device = device

    @torch.no_grad()
    def fill(self, tstamps):
        """SE3 interpolation for a batch of timestamps.

        Args:
            tstamps : list of float timestamps

        Returns:
            SE3 of shape (M,) — world-to-camera poses
        """
        N  = self.video.counter.value
        tt = torch.as_tensor(tstamps, dtype=torch.float64, device=self.device)

        # Cast stored poses to float64 to match tt dtype (avoids lietorch type errors)
        ts = self.video.tstamp[:N].to(self.device)
        Ps = SE3(self.video.poses[:N].to(self.device).double())

        t0 = torch.as_tensor(
            [ts[ts <= t].shape[0] - 1 for t in tstamps],
            dtype=torch.long, device=self.device)
        t0 = torch.clamp(t0, min=0)          # guard: frame before first keyframe
        t1 = torch.where(t0 < N - 1, t0 + 1, t0)

        dt = ts[t1] - ts[t0] + 1e-3
        dP = Ps[t1] * Ps[t0].inv()
        v  = dP.log() / dt.unsqueeze(-1)
        w  = v * (tt - ts[t0]).unsqueeze(-1)
        return SE3.exp(w) * Ps[t0]           # (M,) interpolated w2c poses

    @torch.no_grad()
    def __call__(self, image_stream, batch_size=512):
        """Fill in poses for every frame in image_stream.

        Args:
            image_stream : iterable of (timestamp, image, intrinsic) tuples
                           (image and intrinsic are ignored — only timestamp is used)
        Returns:
            SE3 of shape (T,) — world-to-camera poses for all input frames
        """
        pose_list = []
        tstamps   = []

        for (tstamp, _image, _intrinsic) in image_stream:
            tstamps.append(tstamp)
            if len(tstamps) == batch_size:
                pose_list.append(self.fill(tstamps))
                tstamps = []

        if tstamps:
            pose_list.append(self.fill(tstamps))

        return lietorch.cat(pose_list, 0)

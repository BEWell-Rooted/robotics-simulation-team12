import sys, numpy as np, h5py
f = h5py.File(sys.argv[1], 'r')
demos = list(f['data'].keys())
flips = {}; wneg = {}; tot = 0
ratio = []; lagnorm = []; actnorm = []
rotact = []; rotach = []
for d in demos:
    o = f['data'][d]['obs']; a = f['data'][d]['actions'][:]
    for k in ['eef_quat', 'toy_truck_orientations', 'box_orientations']:
        q = o[k][:].reshape(len(a), -1)
        dots = np.sum(q[1:] * q[:-1], axis=1)
        flips[k] = flips.get(k, 0) + int(np.sum(dots < 0))
        wneg[k] = wneg.get(k, 0) + int(np.sum(q[:, 0] < 0))
    tot += len(a)
    p = o['eef_pos'][:]
    dp = p[1:] - p[:-1]
    ap = a[:-1, :3]
    actnorm.append(np.linalg.norm(ap, axis=1)); lagnorm.append(np.linalg.norm(dp, axis=1))
print('file', sys.argv[1], 'demos', len(demos), 'steps', tot)
print('sign flips (consecutive dot<0):', flips)
print('steps with w<0:', wneg)
an = np.concatenate(actnorm); ln = np.concatenate(lagnorm)
m = an > 1e-4
print('|action pos| median %.4f p90 %.4f ; |achieved dpos| median %.4f p90 %.4f ; median ratio achieved/action %.3f' % (np.median(an), np.percentile(an, 90), np.median(ln), np.percentile(ln, 90), np.median(ln[m] / an[m])))

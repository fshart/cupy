import pytest

import cupy as cp
from cupy.testing import assert_equal, assert_array_almost_equal
from cupy.testing import assert_allclose
from cupyx.scipy.spatial.transform import Rotation, Slerp
# from scipy.stats import special_ortho_group
from itertools import permutations

import pickle
import copy

def test_generic_quat_matrix():
    x = cp.array([[3, 4, 0, 0], [5, 12, 0, 0]])
    r = Rotation.from_quat(x)
    expected_quat = x / cp.array([[5], [13]])
    assert_array_almost_equal(r.as_quat(), expected_quat)


def test_from_single_1d_quaternion():
    x = cp.array([3, 4, 0, 0])
    r = Rotation.from_quat(x)
    expected_quat = x / 5
    assert_array_almost_equal(r.as_quat(), expected_quat)


def test_from_single_2d_quaternion():
    x = cp.array([[3, 4, 0, 0]])
    r = Rotation.from_quat(x)
    expected_quat = x / 5
    assert_array_almost_equal(r.as_quat(), expected_quat)


def test_from_square_quat_matrix():
    # Ensure proper norm array broadcasting
    x = cp.array([
        [3, 0, 0, 4],
        [5, 0, 12, 0],
        [0, 0, 0, 1],
        [-1, -1, -1, 1],
        [0, 0, 0, -1],  # Check double cover
        [-1, -1, -1, -1]  # Check double cover
        ])
    r = Rotation.from_quat(x)
    expected_quat = x / cp.array([[5], [13], [1], [2], [1], [2]])
    assert_array_almost_equal(r.as_quat(), expected_quat)


def test_quat_double_to_canonical_single_cover():
    x = cp.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [-1, -1, -1, -1]
        ])
    r = Rotation.from_quat(x)
    expected_quat = cp.abs(x) / cp.linalg.norm(x, axis=1)[:, None]
    assert_allclose(r.as_quat(canonical=True), expected_quat)


def test_quat_double_cover():
    # See the Rotation.from_quat() docstring for scope of the quaternion
    # double cover property.
    # Check from_quat and as_quat(canonical=False)
    q = cp.array([0, 0, 0, -1])
    r = Rotation.from_quat(q)
    assert_equal(q, r.as_quat(canonical=False))

    # Check composition and inverse
    q = cp.array([1, 0, 0, 1])/cp.sqrt(2)  # 90 deg rotation about x
    r = Rotation.from_quat(q)
    r3 = r*r*r
    assert_allclose(r.as_quat(canonical=False)*cp.sqrt(2),
                    [1, 0, 0, 1])
    assert_allclose(r.inv().as_quat(canonical=False)*cp.sqrt(2),
                    [-1, 0, 0, 1])
    assert_allclose(r3.as_quat(canonical=False)*cp.sqrt(2),
                    [1, 0, 0, -1])
    assert_allclose(r3.inv().as_quat(canonical=False)*cp.sqrt(2),
                    [-1, 0, 0, -1])

    # More sanity checks
    assert_allclose((r*r.inv()).as_quat(canonical=False),
                    [0, 0, 0, 1])
    assert_allclose((r3*r3.inv()).as_quat(canonical=False),
                    [0, 0, 0, 1])
    assert_allclose((r*r3).as_quat(canonical=False),
                    [0, 0, 0, -1])
    assert_allclose((r.inv()*r3.inv()).as_quat(canonical=False),
                    [0, 0, 0, -1])


def test_malformed_1d_from_quat():
    with pytest.raises(ValueError):
        Rotation.from_quat(cp.array([1, 2, 3]))


def test_malformed_2d_from_quat():
    with pytest.raises(ValueError):
        Rotation.from_quat(cp.array([
            [1, 2, 3, 4, 5],
            [4, 5, 6, 7, 8]
            ]))


def test_zero_norms_from_quat():
    x = cp.array([
            [3, 4, 0, 0],
            [0, 0, 0, 0],
            [5, 0, 12, 0]
            ])
    with pytest.raises(ValueError):
        Rotation.from_quat(x)


def test_as_matrix_single_1d_quaternion():
    quat = [0, 0, 0, 1]
    mat = Rotation.from_quat(quat).as_matrix()
    # mat.shape == (3,3) due to 1d input
    assert_array_almost_equal(mat, cp.eye(3))


def test_as_matrix_single_2d_quaternion():
    quat = [[0, 0, 1, 1]]
    mat = Rotation.from_quat(quat).as_matrix()
    assert_equal(mat.shape, (1, 3, 3))
    expected_mat = cp.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
        ])
    assert_array_almost_equal(mat[0], expected_mat)


def test_as_matrix_from_square_input():
    quats = [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, -1]
            ]
    mat = Rotation.from_quat(quats).as_matrix()
    assert_equal(mat.shape, (4, 3, 3))

    expected0 = cp.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
        ])
    assert_array_almost_equal(mat[0], expected0)

    expected1 = cp.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
        ])
    assert_array_almost_equal(mat[1], expected1)

    assert_array_almost_equal(mat[2], cp.eye(3))
    assert_array_almost_equal(mat[3], cp.eye(3))


def test_as_matrix_from_generic_input():
    quats = [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 2, 3, 4]
            ]
    mat = Rotation.from_quat(quats).as_matrix()
    assert_equal(mat.shape, (3, 3, 3))

    expected0 = cp.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
        ])
    assert_array_almost_equal(mat[0], expected0)

    expected1 = cp.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
        ])
    assert_array_almost_equal(mat[1], expected1)

    expected2 = cp.array([
        [0.4, -2, 2.2],
        [2.8, 1, 0.4],
        [-1, 2, 2]
        ]) / 3
    assert_array_almost_equal(mat[2], expected2)


def test_from_single_2d_matrix():
    mat = [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
            ]
    expected_quat = [0.5, 0.5, 0.5, 0.5]
    assert_array_almost_equal(
            Rotation.from_matrix(mat).as_quat(),
            expected_quat)


def test_from_single_3d_matrix():
    mat = cp.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
        ]).reshape((1, 3, 3))
    expected_quat = cp.array([0.5, 0.5, 0.5, 0.5]).reshape((1, 4))
    assert_array_almost_equal(
            Rotation.from_matrix(mat).as_quat(),
            expected_quat)


def test_from_matrix_calculation():
    expected_quat = cp.array([1, 1, 6, 1]) / cp.sqrt(39)
    mat = cp.array([
            [-0.8974359, -0.2564103, 0.3589744],
            [0.3589744, -0.8974359, 0.2564103],
            [0.2564103, 0.3589744, 0.8974359]
            ])
    assert_array_almost_equal(
            Rotation.from_matrix(mat).as_quat(),
            expected_quat)
    assert_array_almost_equal(
            Rotation.from_matrix(mat.reshape((1, 3, 3))).as_quat(),
            expected_quat.reshape((1, 4)))


# def test_matrix_calculation_pipeline():
#     mat = special_ortho_group.rvs(3, size=10, random_state=0)
#     assert_array_almost_equal(Rotation.from_matrix(mat).as_matrix(), mat)


def test_from_matrix_ortho_output():
    rnd = cp.random.RandomState(0)
    mat = rnd.random_sample((100, 3, 3))
    ortho_mat = Rotation.from_matrix(mat).as_matrix()

    mult_result = cp.einsum('...ij,...jk->...ik', ortho_mat,
                            ortho_mat.transpose((0, 2, 1)))

    eye3d = cp.zeros((100, 3, 3))
    for i in range(3):
        eye3d[:, i, i] = 1.0

    assert_array_almost_equal(mult_result, eye3d)


def test_from_1d_single_rotvec():
    rotvec = [1, 0, 0]
    expected_quat = cp.array([0.4794255, 0, 0, 0.8775826])
    result = Rotation.from_rotvec(rotvec)
    assert_array_almost_equal(result.as_quat(), expected_quat)


def test_from_2d_single_rotvec():
    rotvec = [[1, 0, 0]]
    expected_quat = cp.array([[0.4794255, 0, 0, 0.8775826]])
    result = Rotation.from_rotvec(rotvec)
    assert_array_almost_equal(result.as_quat(), expected_quat)


def test_from_generic_rotvec():
    rotvec = [
            [1, 2, 2],
            [1, -1, 0.5],
            [0, 0, 0]
            ]
    expected_quat = cp.array([
        [0.3324983, 0.6649967, 0.6649967, 0.0707372],
        [0.4544258, -0.4544258, 0.2272129, 0.7316889],
        [0, 0, 0, 1]
        ])
    assert_array_almost_equal(
            Rotation.from_rotvec(rotvec).as_quat(),
            expected_quat)


def test_from_rotvec_small_angle():
    rotvec = cp.array([
        [5e-4 / cp.sqrt(3), -5e-4 / cp.sqrt(3), 5e-4 / cp.sqrt(3)],
        [0.2, 0.3, 0.4],
        [0, 0, 0]
        ])

    quat = Rotation.from_rotvec(rotvec).as_quat()
    # cos(theta/2) ~~ 1 for small theta
    assert_allclose(quat[0, 3], 1)
    # sin(theta/2) / theta ~~ 0.5 for small theta
    assert_allclose(quat[0, :3], rotvec[0] * 0.5)

    assert_allclose(quat[1, 3], 0.9639685)
    assert_allclose(
            quat[1, :3],
            cp.array([
                0.09879603932153465,
                0.14819405898230198,
                0.19759207864306931
                ]))

    assert_equal(quat[2], cp.array([0, 0, 0, 1]))


def test_degrees_from_rotvec():
    rotvec1 = [1.0 / cp.cbrt(3), 1.0 / cp.cbrt(3), 1.0 / cp.cbrt(3)]
    rot1 = Rotation.from_rotvec(rotvec1, degrees=True)
    quat1 = rot1.as_quat()

    rotvec2 = cp.deg2rad(rotvec1)
    rot2 = Rotation.from_rotvec(rotvec2)
    quat2 = rot2.as_quat()

    assert_allclose(quat1, quat2)


def test_malformed_1d_from_rotvec():
    with pytest.raises(ValueError, match='Expected `rot_vec` to have shape'):
        Rotation.from_rotvec([1, 2])


def test_malformed_2d_from_rotvec():
    with pytest.raises(ValueError, match='Expected `rot_vec` to have shape'):
        Rotation.from_rotvec([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
            ])


def test_as_generic_rotvec():
    quat = cp.array([
            [1, 2, -1, 0.5],
            [1, -1, 1, 0.0003],
            [0, 0, 0, 1]
            ])
    quat /= cp.linalg.norm(quat, axis=1)[:, None]

    rotvec = Rotation.from_quat(quat).as_rotvec()
    angle = cp.linalg.norm(rotvec, axis=1)

    assert_allclose(quat[:, 3], cp.cos(angle/2))
    assert_allclose(cp.cross(rotvec, quat[:, :3]), cp.zeros((3, 3)))


def test_as_rotvec_single_1d_input():
    quat = cp.array([1, 2, -3, 2])
    expected_rotvec = cp.array([0.5772381, 1.1544763, -1.7317144])

    actual_rotvec = Rotation.from_quat(quat).as_rotvec()

    assert_equal(actual_rotvec.shape, (3,))
    assert_allclose(actual_rotvec, expected_rotvec)


def test_as_rotvec_single_2d_input():
    quat = cp.array([[1, 2, -3, 2]])
    expected_rotvec = cp.array([[0.5772381, 1.1544763, -1.7317144]])

    actual_rotvec = Rotation.from_quat(quat).as_rotvec()

    assert_equal(actual_rotvec.shape, (1, 3))
    assert_allclose(actual_rotvec, expected_rotvec)


def test_as_rotvec_degrees():
    # x->y, y->z, z->x
    mat = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    rot = Rotation.from_matrix(mat)
    rotvec = rot.as_rotvec(degrees=True)
    angle = cp.linalg.norm(rotvec)
    assert_allclose(angle, 120.0)
    assert_allclose(rotvec[0], rotvec[1])
    assert_allclose(rotvec[1], rotvec[2])


def test_rotvec_calc_pipeline():
    # Include small angles
    rotvec = cp.array([
        [0, 0, 0],
        [1, -1, 2],
        [-3e-4, 3.5e-4, 7.5e-5]
        ])
    assert_allclose(Rotation.from_rotvec(rotvec).as_rotvec(), rotvec)
    assert_allclose(Rotation.from_rotvec(rotvec, degrees=True).as_rotvec(degrees=True), rotvec)


def test_from_1d_single_mrp():
    mrp = [0, 0, 1.0]
    expected_quat = cp.array([0, 0, 1, 0])
    result = Rotation.from_mrp(mrp)
    assert_array_almost_equal(result.as_quat(), expected_quat)


def test_from_2d_single_mrp():
    mrp = [[0, 0, 1.0]]
    expected_quat = cp.array([[0, 0, 1, 0]])
    result = Rotation.from_mrp(mrp)
    assert_array_almost_equal(result.as_quat(), expected_quat)


def test_from_generic_mrp():
    mrp = cp.array([
        [1, 2, 2],
        [1, -1, 0.5],
        [0, 0, 0]])
    expected_quat = cp.array([
        [0.2, 0.4, 0.4, -0.8],
        [0.61538462, -0.61538462, 0.30769231, -0.38461538],
        [0, 0, 0, 1]])
    assert_array_almost_equal(Rotation.from_mrp(mrp).as_quat(), expected_quat)


def test_malformed_1d_from_mrp():
    with pytest.raises(ValueError, match='Expected `mrp` to have shape'):
        Rotation.from_mrp([1, 2])


def test_malformed_2d_from_mrp():
    with pytest.raises(ValueError, match='Expected `mrp` to have shape'):
        Rotation.from_mrp([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
            ])


def test_as_generic_mrp():
    quat = cp.array([
        [1, 2, -1, 0.5],
        [1, -1, 1, 0.0003],
        [0, 0, 0, 1]])
    quat /= cp.linalg.norm(quat, axis=1)[:, None]

    expected_mrp = cp.array([
        [0.33333333, 0.66666667, -0.33333333],
        [0.57725028, -0.57725028, 0.57725028],
        [0, 0, 0]])
    assert_array_almost_equal(Rotation.from_quat(quat).as_mrp(), expected_mrp)

def test_past_180_degree_rotation():
    # ensure that a > 180 degree rotation is returned as a <180 rotation in MRPs
    # in this case 270 should be returned as -90
    expected_mrp = cp.array([-cp.tan(np.pi/2/4), 0.0, 0])
    assert_array_almost_equal(Rotation.from_euler('xyz', [270, 0, 0], degrees=True).as_mrp(), expected_mrp)


def test_as_mrp_single_1d_input():
    quat = cp.array([1, 2, -3, 2])
    expected_mrp = cp.array([0.16018862, 0.32037724, -0.48056586])

    actual_mrp = Rotation.from_quat(quat).as_mrp()

    assert_equal(actual_mrp.shape, (3,))
    assert_allclose(actual_mrp, expected_mrp)


def test_as_mrp_single_2d_input():
    quat = cp.array([[1, 2, -3, 2]])
    expected_mrp = cp.array([[0.16018862, 0.32037724, -0.48056586]])

    actual_mrp = Rotation.from_quat(quat).as_mrp()

    assert_equal(actual_mrp.shape, (1, 3))
    assert_allclose(actual_mrp, expected_mrp)


def test_mrp_calc_pipeline():
    actual_mrp = cp.array([
        [0, 0, 0],
        [1, -1, 2],
        [0.41421356, 0, 0],
        [0.1, 0.2, 0.1]])
    expected_mrp = cp.array([
        [0, 0, 0],
        [-0.16666667, 0.16666667, -0.33333333],
        [0.41421356, 0, 0],
        [0.1, 0.2, 0.1]])
    assert_allclose(Rotation.from_mrp(actual_mrp).as_mrp(), expected_mrp)


def test_from_euler_single_rotation():
    quat = Rotation.from_euler('z', 90, degrees=True).as_quat()
    expected_quat = cp.array([0, 0, 1, 1]) / cp.sqrt(2)
    assert_allclose(quat, expected_quat)


def test_single_intrinsic_extrinsic_rotation():
    extrinsic = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    intrinsic = Rotation.from_euler('Z', 90, degrees=True).as_matrix()
    assert_allclose(extrinsic, intrinsic)


def test_from_euler_rotation_order():
    # Intrinsic rotation is same as extrinsic with order reversed
    rnd = cp.random.RandomState(0)
    a = rnd.randint(low=0, high=180, size=(6, 3))
    b = a[:, ::-1]
    x = Rotation.from_euler('xyz', a, degrees=True).as_quat()
    y = Rotation.from_euler('ZYX', b, degrees=True).as_quat()
    assert_allclose(x, y)


def test_from_euler_elementary_extrinsic_rotation():
    # Simple test to check if extrinsic rotations are implemented correctly
    mat = Rotation.from_euler('zx', [90, 90], degrees=True).as_matrix()
    expected_mat = cp.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ])
    assert_array_almost_equal(mat, expected_mat)


def test_from_euler_intrinsic_rotation_312():
    angles = [
        [30, 60, 45],
        [30, 60, 30],
        [45, 30, 60]
        ]
    mat = Rotation.from_euler('ZXY', angles, degrees=True).as_matrix()

    assert_array_almost_equal(mat[0], cp.array([
        [0.3061862, -0.2500000, 0.9185587],
        [0.8838835, 0.4330127, -0.1767767],
        [-0.3535534, 0.8660254, 0.3535534]
    ]))

    assert_array_almost_equal(mat[1], cp.array([
        [0.5334936, -0.2500000, 0.8080127],
        [0.8080127, 0.4330127, -0.3995191],
        [-0.2500000, 0.8660254, 0.4330127]
    ]))

    assert_array_almost_equal(mat[2], cp.array([
        [0.0473672, -0.6123725, 0.7891491],
        [0.6597396, 0.6123725, 0.4355958],
        [-0.7500000, 0.5000000, 0.4330127]
    ]))


def test_from_euler_intrinsic_rotation_313():
    angles = [
        [30, 60, 45],
        [30, 60, 30],
        [45, 30, 60]
        ]
    mat = Rotation.from_euler('ZXZ', angles, degrees=True).as_matrix()

    assert_array_almost_equal(mat[0], cp.array([
        [0.43559574, -0.78914913, 0.4330127],
        [0.65973961, -0.04736717, -0.750000],
        [0.61237244, 0.61237244, 0.500000]
    ]))

    assert_array_almost_equal(mat[1], cp.array([
        [0.6250000, -0.64951905, 0.4330127],
        [0.64951905, 0.1250000, -0.750000],
        [0.4330127, 0.750000, 0.500000]
    ]))

    assert_array_almost_equal(mat[2], cp.array([
        [-0.1767767, -0.91855865, 0.35355339],
        [0.88388348, -0.30618622, -0.35355339],
        [0.4330127, 0.25000000, 0.8660254]
    ]))


def test_from_euler_extrinsic_rotation_312():
    angles = [
        [30, 60, 45],
        [30, 60, 30],
        [45, 30, 60]
        ]
    mat = Rotation.from_euler('zxy', angles, degrees=True).as_matrix()

    assert_array_almost_equal(mat[0], cp.array([
        [0.91855865, 0.1767767, 0.35355339],
        [0.25000000, 0.4330127, -0.8660254],
        [-0.30618622, 0.88388348, 0.35355339]
    ]))

    assert_array_almost_equal(mat[1], cp.array([
        [0.96650635, -0.0580127, 0.2500000],
        [0.25000000, 0.4330127, -0.8660254],
        [-0.0580127, 0.89951905, 0.4330127]
    ]))

    assert_array_almost_equal(mat[2], cp.array([
        [0.65973961, -0.04736717, 0.7500000],
        [0.61237244, 0.61237244, -0.5000000],
        [-0.43559574, 0.78914913, 0.4330127]
    ]))


def test_from_euler_extrinsic_rotation_313():
    angles = [
        [30, 60, 45],
        [30, 60, 30],
        [45, 30, 60]
        ]
    mat = Rotation.from_euler('zxz', angles, degrees=True).as_matrix()

    assert_array_almost_equal(mat[0], cp.array([
        [0.43559574, -0.65973961, 0.61237244],
        [0.78914913, -0.04736717, -0.61237244],
        [0.4330127, 0.75000000, 0.500000]
    ]))

    assert_array_almost_equal(mat[1], cp.array([
        [0.62500000, -0.64951905, 0.4330127],
        [0.64951905, 0.12500000, -0.750000],
        [0.4330127, 0.75000000, 0.500000]
    ]))

    assert_array_almost_equal(mat[2], cp.array([
        [-0.1767767, -0.88388348, 0.4330127],
        [0.91855865, -0.30618622, -0.250000],
        [0.35355339, 0.35355339, 0.8660254]
    ]))


def test_as_euler_asymmetric_axes():
    # helper function for mean error tests
    def test_stats(error, mean_max, rms_max):
        mean = cp.mean(error, axis=0)
        std = cp.std(error, axis=0)
        rms = cp.hypot(mean, std)
        assert cp.all(cp.abs(mean) < mean_max)
        assert cp.all(rms < rms_max)

    rnd = cp.random.RandomState(0)
    n = 1000
    angles = cp.empty((n, 3))
    angles[:, 0] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))
    angles[:, 1] = rnd.uniform(low=-np.pi / 2, high=np.pi / 2, size=(n,))
    angles[:, 2] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))

    for seq_tuple in permutations('xyz'):
        # Extrinsic rotations
        seq = ''.join(seq_tuple)
        rotation = Rotation.from_euler(seq, angles)
        angles_quat = rotation.as_euler(seq)
        angles_mat = rotation._as_euler_from_matrix(seq)
        assert_allclose(angles, angles_quat, atol=0, rtol=1e-12)
        assert_allclose(angles, angles_mat, atol=0, rtol=1e-12)
        test_stats(angles_quat - angles, 1e-15, 1e-14)
        test_stats(angles_mat - angles, 1e-15, 1e-14)

        # Intrinsic rotations
        seq = seq.upper()
        rotation = Rotation.from_euler(seq, angles)
        angles_quat = rotation.as_euler(seq)
        angles_mat = rotation._as_euler_from_matrix(seq)
        assert_allclose(angles, angles_quat, atol=0, rtol=1e-12)
        assert_allclose(angles, angles_mat, atol=0, rtol=1e-12)
        test_stats(angles_quat - angles, 1e-15, 1e-14)
        test_stats(angles_mat - angles, 1e-15, 1e-14)


def test_as_euler_symmetric_axes():
    # helper function for mean error tests
    def test_stats(error, mean_max, rms_max):
        mean = cp.mean(error, axis=0)
        std = cp.std(error, axis=0)
        rms = cp.hypot(mean, std)
        assert cp.all(cp.abs(mean) < mean_max)
        assert cp.all(rms < rms_max)

    rnd = cp.random.RandomState(0)
    n = 1000
    angles = cp.empty((n, 3))
    angles[:, 0] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))
    angles[:, 1] = rnd.uniform(low=0, high=np.pi, size=(n,))
    angles[:, 2] = rnd.uniform(low=-np.pi, high=np.pi, size=(n,))

    for seq_tuple in permutations('xyz'):
        # Extrinsic rotations
        seq = ''.join([seq_tuple[0], seq_tuple[1], seq_tuple[0]])
        rotation = Rotation.from_euler(seq, angles)
        angles_quat = rotation.as_euler(seq)
        angles_mat = rotation._as_euler_from_matrix(seq)
        assert_allclose(angles, angles_quat, atol=0, rtol=1e-13)
        assert_allclose(angles, angles_mat, atol=0, rtol=1e-9)
        test_stats(angles_quat - angles, 1e-16, 1e-14)
        test_stats(angles_mat - angles, 1e-15, 1e-13)

        # Intrinsic rotations
        seq = seq.upper()
        rotation = Rotation.from_euler(seq, angles)
        angles_quat = rotation.as_euler(seq)
        angles_mat = rotation._as_euler_from_matrix(seq)
        assert_allclose(angles, angles_quat, atol=0, rtol=1e-13)
        assert_allclose(angles, angles_mat, atol=0, rtol=1e-9)
        test_stats(angles_quat - angles, 1e-16, 1e-14)
        test_stats(angles_mat - angles, 1e-15, 1e-13)


def test_as_euler_degenerate_asymmetric_axes():
    # Since we cannot check for angle equality, we check for rotation matrix
    # equality
    angles = cp.array([
        [45, 90, 35],
        [35, -90, 20],
        [35, 90, 25],
        [25, -90, 15]])

    with pytest.warns(UserWarning, match="Gimbal lock"):
        for seq_tuple in permutations('xyz'):
            # Extrinsic rotations
            seq = ''.join(seq_tuple)
            rotation = Rotation.from_euler(seq, angles, degrees=True)
            mat_expected = rotation.as_matrix()

            angle_estimates = rotation.as_euler(seq, degrees=True)
            mat_estimated = Rotation.from_euler(
                seq, angle_estimates, degrees=True
                ).as_matrix()

            assert_array_almost_equal(mat_expected, mat_estimated)

            # Intrinsic rotations
            seq = seq.upper()
            rotation = Rotation.from_euler(seq, angles, degrees=True)
            mat_expected = rotation.as_matrix()

            angle_estimates = rotation.as_euler(seq, degrees=True)
            mat_estimated = Rotation.from_euler(
                seq, angle_estimates, degrees=True
                ).as_matrix()

            assert_array_almost_equal(mat_expected, mat_estimated)


def test_as_euler_degenerate_symmetric_axes():
    # Since we cannot check for angle equality, we check for rotation matrix
    # equality
    angles = cp.array([
        [15, 0, 60],
        [35, 0, 75],
        [60, 180, 35],
        [15, -180, 25]])

    with pytest.warns(UserWarning, match="Gimbal lock"):
        for seq_tuple in permutations('xyz'):
            # Extrinsic rotations
            seq = ''.join([seq_tuple[0], seq_tuple[1], seq_tuple[0]])
            rotation = Rotation.from_euler(seq, angles, degrees=True)
            mat_expected = rotation.as_matrix()

            angle_estimates = rotation.as_euler(seq, degrees=True)
            mat_estimated = Rotation.from_euler(
                seq, angle_estimates, degrees=True
                ).as_matrix()

            assert_array_almost_equal(mat_expected, mat_estimated)

            # Intrinsic rotations
            seq = seq.upper()
            rotation = Rotation.from_euler(seq, angles, degrees=True)
            mat_expected = rotation.as_matrix()

            angle_estimates = rotation.as_euler(seq, degrees=True)
            mat_estimated = Rotation.from_euler(
                seq, angle_estimates, degrees=True
                ).as_matrix()

            assert_array_almost_equal(mat_expected, mat_estimated)


def test_as_euler_degenerate_compare_algorithms():
    # this test makes sure that both algorithms are doing the same choices
    # in degenerate cases

    # asymmetric axes
    angles = cp.array([
        [45, 90, 35],
        [35, -90, 20],
        [35, 90, 25],
        [25, -90, 15]])

    with pytest.warns(UserWarning, match="Gimbal lock"):
        for seq_tuple in permutations('xyz'):
            # Extrinsic rotations
            seq = ''.join(seq_tuple)
            rot = Rotation.from_euler(seq, angles, degrees=True)
            estimates_matrix = rot._as_euler_from_matrix(seq, degrees=True)
            estimates_quat = rot.as_euler(seq, degrees=True)
            assert_allclose(estimates_matrix[:, [0, 2]],
                            estimates_quat[:, [0, 2]], atol=0, rtol=1e-12)
            assert_allclose(estimates_matrix[:, 1], estimates_quat[:, 1],
                            atol=0, rtol=1e-7)

            # Intrinsic rotations
            seq = seq.upper()
            rot = Rotation.from_euler(seq, angles, degrees=True)
            estimates_matrix = rot._as_euler_from_matrix(seq, degrees=True)
            estimates_quat = rot.as_euler(seq, degrees=True)
            assert_allclose(estimates_matrix[:, [0, 2]],
                            estimates_quat[:, [0, 2]], atol=0, rtol=1e-12)
            assert_allclose(estimates_matrix[:, 1], estimates_quat[:, 1],
                            atol=0, rtol=1e-7)
    # symmetric axes
    # Absolute error tolerance must be looser to directly compare the results
    # from both algorithms, because of numerical loss of precision for the
    # method _as_euler_from_matrix near a zero angle value

    angles = cp.array([
        [15, 0, 60],
        [35, 0, 75],
        [60, 180, 35],
        [15, -180, 25]])

    idx = angles[:, 1] == 0  # find problematic angles indices

    with pytest.warns(UserWarning, match="Gimbal lock"):
        for seq_tuple in permutations('xyz'):
            # Extrinsic rotations
            seq = ''.join([seq_tuple[0], seq_tuple[1], seq_tuple[0]])
            rot = Rotation.from_euler(seq, angles, degrees=True)
            estimates_matrix = rot._as_euler_from_matrix(seq, degrees=True)
            estimates_quat = rot.as_euler(seq, degrees=True)
            assert_allclose(estimates_matrix[:, [0, 2]],
                            estimates_quat[:, [0, 2]], atol=0, rtol=1e-12)

            assert_allclose(estimates_matrix[~idx, 1], estimates_quat[~idx, 1],
                            atol=0, rtol=1e-7)

            assert_allclose(estimates_matrix[idx, 1], estimates_quat[idx, 1],
                            atol=1e-6)  # problematic, angles[1] = 0

            # Intrinsic rotations
            seq = seq.upper()
            rot = Rotation.from_euler(seq, angles, degrees=True)
            estimates_matrix = rot._as_euler_from_matrix(seq, degrees=True)
            estimates_quat = rot.as_euler(seq, degrees=True)
            assert_allclose(estimates_matrix[:, [0, 2]],
                            estimates_quat[:, [0, 2]], atol=0, rtol=1e-12)

            assert_allclose(estimates_matrix[~idx, 1], estimates_quat[~idx, 1],
                            atol=0, rtol=1e-7)

            assert_allclose(estimates_matrix[idx, 1], estimates_quat[idx, 1],
                            atol=1e-6)  # problematic, angles[1] = 0


def test_inv():
    rnd = cp.random.RandomState(0)
    n = 10
    p = Rotation.random(num=n, random_state=rnd)
    q = p.inv()

    p_mat = p.as_matrix()
    q_mat = q.as_matrix()
    result1 = cp.einsum('...ij,...jk->...ik', p_mat, q_mat)
    result2 = cp.einsum('...ij,...jk->...ik', q_mat, p_mat)

    eye3d = cp.empty((n, 3, 3))
    eye3d[:] = cp.eye(3)

    assert_array_almost_equal(result1, eye3d)
    assert_array_almost_equal(result2, eye3d)


def test_inv_single_rotation():
    rnd = cp.random.RandomState(0)
    p = Rotation.random(random_state=rnd)
    q = p.inv()

    p_mat = p.as_matrix()
    q_mat = q.as_matrix()
    res1 = cp.dot(p_mat, q_mat)
    res2 = cp.dot(q_mat, p_mat)

    eye = cp.eye(3)

    assert_array_almost_equal(res1, eye)
    assert_array_almost_equal(res2, eye)

    x = Rotation.random(num=1, random_state=rnd)
    y = x.inv()

    x_matrix = x.as_matrix()
    y_matrix = y.as_matrix()
    result1 = cp.einsum('...ij,...jk->...ik', x_matrix, y_matrix)
    result2 = cp.einsum('...ij,...jk->...ik', y_matrix, x_matrix)

    eye3d = cp.empty((1, 3, 3))
    eye3d[:] = cp.eye(3)

    assert_array_almost_equal(result1, eye3d)
    assert_array_almost_equal(result2, eye3d)


def test_identity_magnitude():
    n = 10
    assert_allclose(Rotation.identity(n).magnitude(), 0)
    assert_allclose(Rotation.identity(n).inv().magnitude(), 0)


def test_single_identity_magnitude():
    assert Rotation.identity().magnitude() == 0
    assert Rotation.identity().inv().magnitude() == 0


def test_identity_invariance():
    n = 10
    p = Rotation.random(n, random_state=0)

    result = p * Rotation.identity(n)
    assert_array_almost_equal(p.as_quat(), result.as_quat())

    result = result * p.inv()
    assert_array_almost_equal(result.magnitude(), cp.zeros(n))


def test_single_identity_invariance():
    n = 10
    p = Rotation.random(n, random_state=0)

    result = p * Rotation.identity()
    assert_array_almost_equal(p.as_quat(), result.as_quat())

    result = result * p.inv()
    assert_array_almost_equal(result.magnitude(), cp.zeros(n))


def test_magnitude():
    r = Rotation.from_quat(cp.eye(4))
    result = r.magnitude()
    assert_array_almost_equal(result, [np.pi, np.pi, np.pi, 0])

    r = Rotation.from_quat(-cp.eye(4))
    result = r.magnitude()
    assert_array_almost_equal(result, [np.pi, np.pi, np.pi, 0])


def test_magnitude_single_rotation():
    r = Rotation.from_quat(cp.eye(4))
    result1 = r[0].magnitude()
    assert_allclose(result1, np.pi)

    result2 = r[3].magnitude()
    assert_allclose(result2, 0)


def test_approx_equal():
    rng = cp.random.RandomState(0)
    p = Rotation.random(10, random_state=rng)
    q = Rotation.random(10, random_state=rng)
    r = p * q.inv()
    r_mag = r.magnitude()
    atol = cp.median(r_mag)  # ensure we get mix of Trues and Falses
    assert_equal(p.approx_equal(q, atol), (r_mag < atol))


def test_approx_equal_single_rotation():
    # also tests passing single argument to approx_equal
    p = Rotation.from_rotvec([0, 0, 1e-9])  # less than default atol of 1e-8
    q = Rotation.from_quat(cp.eye(4))
    assert p.approx_equal(q[3])
    assert not p.approx_equal(q[0])

    # test passing atol and using degrees
    assert not p.approx_equal(q[3], atol=1e-10)
    assert not p.approx_equal(q[3], atol=1e-8, degrees=True)
    with pytest.warns(UserWarning, match="atol must be set"):
        assert p.approx_equal(q[3], degrees=True)


def test_mean():
    axes = cp.concatenate((-cp.eye(3), cp.eye(3)))
    thetas = cp.linspace(0, np.pi / 2, 100)
    for t in thetas:
        r = Rotation.from_rotvec(t * axes)
        assert_allclose(r.mean().magnitude(), 0, atol=1E-10)


def test_weighted_mean():
    # test that doubling a weight is equivalent to including a rotation twice.
    axes = cp.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    thetas = cp.linspace(0, np.pi / 2, 100)
    for t in thetas:
        rw = Rotation.from_rotvec(t * axes[:2])
        mw = rw.mean(weights=[1, 2])

        r = Rotation.from_rotvec(t * axes)
        m = r.mean()
        assert_allclose((m * mw.inv()).magnitude(), 0, atol=1E-10)


def test_mean_invalid_weights():
    with pytest.raises(ValueError, match="non-negative"):
        r = Rotation.from_quat(cp.eye(4))
        r.mean(weights=-cp.ones(4))


def test_reduction_no_indices():
    result = Rotation.identity().reduce(return_indices=False)
    assert isinstance(result, Rotation)


def test_reduction_none_indices():
    result = Rotation.identity().reduce(return_indices=True)
    assert type(result) == tuple
    assert len(result) == 3

    reduced, left_best, right_best = result
    assert left_best is None
    assert right_best is None


def test_reduction_scalar_calculation():
    rng = cp.random.RandomState(0)
    l = Rotation.random(5, random_state=rng)
    r = Rotation.random(10, random_state=rng)
    p = Rotation.random(7, random_state=rng)
    reduced, left_best, right_best = p.reduce(l, r, return_indices=True)

    # Loop implementation of the vectorized calculation in Rotation.reduce
    scalars = cp.zeros((len(l), len(p), len(r)))
    for i, li in enumerate(l):
        for j, pj in enumerate(p):
            for k, rk in enumerate(r):
                scalars[i, j, k] = cp.abs((li * pj * rk).as_quat()[3])
    scalars = cp.reshape(cp.moveaxis(scalars, 1, 0), (scalars.shape[1], -1))

    max_ind = cp.argmax(cp.reshape(scalars, (len(p), -1)), axis=1)
    left_best_check = max_ind // len(r)
    right_best_check = max_ind % len(r)
    assert (left_best == left_best_check).all()
    assert (right_best == right_best_check).all()

    reduced_check = l[left_best_check] * p * r[right_best_check]
    mag = (reduced.inv() * reduced_check).magnitude()
    assert_array_almost_equal(mag, cp.zeros(len(p)))


def test_apply_single_rotation_single_point():
    mat = cp.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    r_1d = Rotation.from_matrix(mat)
    r_2d = Rotation.from_matrix(cp.expand_dims(mat, axis=0))

    v_1d = cp.array([1, 2, 3])
    v_2d = cp.expand_dims(v_1d, axis=0)
    v1d_rotated = cp.array([-2, 1, 3])
    v2d_rotated = cp.expand_dims(v1d_rotated, axis=0)

    assert_allclose(r_1d.apply(v_1d), v1d_rotated)
    assert_allclose(r_1d.apply(v_2d), v2d_rotated)
    assert_allclose(r_2d.apply(v_1d), v2d_rotated)
    assert_allclose(r_2d.apply(v_2d), v2d_rotated)

    v1d_inverse = cp.array([2, -1, 3])
    v2d_inverse = cp.expand_dims(v1d_inverse, axis=0)

    assert_allclose(r_1d.apply(v_1d, inverse=True), v1d_inverse)
    assert_allclose(r_1d.apply(v_2d, inverse=True), v2d_inverse)
    assert_allclose(r_2d.apply(v_1d, inverse=True), v2d_inverse)
    assert_allclose(r_2d.apply(v_2d, inverse=True), v2d_inverse)


def test_apply_single_rotation_multiple_points():
    mat = cp.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    r1 = Rotation.from_matrix(mat)
    r2 = Rotation.from_matrix(cp.expand_dims(mat, axis=0))

    v = cp.array([[1, 2, 3], [4, 5, 6]])
    v_rotated = cp.array([[-2, 1, 3], [-5, 4, 6]])

    assert_allclose(r1.apply(v), v_rotated)
    assert_allclose(r2.apply(v), v_rotated)

    v_inverse = cp.array([[2, -1, 3], [5, -4, 6]])

    assert_allclose(r1.apply(v, inverse=True), v_inverse)
    assert_allclose(r2.apply(v, inverse=True), v_inverse)


def test_apply_multiple_rotations_single_point():
    mat = cp.empty((2, 3, 3))
    mat[0] = cp.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    mat[1] = cp.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    r = Rotation.from_matrix(mat)

    v1 = cp.array([1, 2, 3])
    v2 = cp.expand_dims(v1, axis=0)

    v_rotated = cp.array([[-2, 1, 3], [1, -3, 2]])

    assert_allclose(r.apply(v1), v_rotated)
    assert_allclose(r.apply(v2), v_rotated)

    v_inverse = cp.array([[2, -1, 3], [1, 3, -2]])

    assert_allclose(r.apply(v1, inverse=True), v_inverse)
    assert_allclose(r.apply(v2, inverse=True), v_inverse)


def test_apply_multiple_rotations_multiple_points():
    mat = cp.empty((2, 3, 3))
    mat[0] = cp.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    mat[1] = cp.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    r = Rotation.from_matrix(mat)

    v = cp.array([[1, 2, 3], [4, 5, 6]])
    v_rotated = cp.array([[-2, 1, 3], [4, -6, 5]])
    assert_allclose(r.apply(v), v_rotated)

    v_inverse = cp.array([[2, -1, 3], [4, 6, -5]])
    assert_allclose(r.apply(v, inverse=True), v_inverse)


def test_getitem():
    mat = cp.empty((2, 3, 3))
    mat[0] = cp.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    mat[1] = cp.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    r = Rotation.from_matrix(mat)

    assert_allclose(r[0].as_matrix(), mat[0], atol=1e-15)
    assert_allclose(r[1].as_matrix(), mat[1], atol=1e-15)
    assert_allclose(r[:-1].as_matrix(), cp.expand_dims(mat[0], axis=0), atol=1e-15)


def test_getitem_single():
    with pytest.raises(TypeError, match='not subscriptable'):
        Rotation.identity()[0]


def test_setitem_single():
    r = Rotation.identity()
    with pytest.raises(TypeError, match='not subscriptable'):
        r[0] = Rotation.identity()


def test_setitem_slice():
    rng = cp.random.RandomState(seed=0)
    r1 = Rotation.random(10, random_state=rng)
    r2 = Rotation.random(5, random_state=rng)
    r1[1:6] = r2
    assert_equal(r1[1:6].as_quat(), r2.as_quat())


def test_setitem_integer():
    rng = cp.random.RandomState(seed=0)
    r1 = Rotation.random(10, random_state=rng)
    r2 = Rotation.random(random_state=rng)
    r1[1] = r2
    assert_equal(r1[1].as_quat(), r2.as_quat())


def test_setitem_wrong_type():
    r = Rotation.random(10, random_state=0)
    with pytest.raises(TypeError, match='Rotation object'):
        r[0] = 1


def test_n_rotations():
    mat = cp.empty((2, 3, 3))
    mat[0] = cp.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    mat[1] = cp.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    r = Rotation.from_matrix(mat)

    assert_equal(len(r), 2)
    assert_equal(len(r[:-1]), 1)


def test_align_vectors_no_rotation():
    x = cp.array([[1, 2, 3], [4, 5, 6]])
    y = x.copy()

    r, rmsd = Rotation.align_vectors(x, y)
    assert_array_almost_equal(r.as_matrix(), cp.eye(3))
    assert_allclose(rmsd, 0, atol=1e-6)


def test_align_vectors_no_noise():
    rnd = cp.random.RandomState(0)
    c = Rotation.random(random_state=rnd)
    b = rnd.normal(size=(5, 3))
    a = c.apply(b)

    est, rmsd = Rotation.align_vectors(a, b)
    assert_allclose(c.as_quat(), est.as_quat())
    assert_allclose(rmsd, 0, atol=1e-7)


def test_align_vectors_improper_rotation():
    # Tests correct logic for issue #10444
    x = cp.array([[0.89299824, -0.44372674, 0.0752378],
                  [0.60221789, -0.47564102, -0.6411702]])
    y = cp.array([[0.02386536, -0.82176463, 0.5693271],
                  [-0.27654929, -0.95191427, -0.1318321]])

    est, rmsd = Rotation.align_vectors(x, y)
    assert_allclose(x, est.apply(y), atol=1e-6)
    assert_allclose(rmsd, 0, atol=1e-7)


def test_align_vectors_scaled_weights():
    rng = cp.random.RandomState(0)
    c = Rotation.random(random_state=rng)
    b = rng.normal(size=(5, 3))
    a = c.apply(b)

    est1, rmsd1, cov1 = Rotation.align_vectors(a, b, cp.ones(5), True)
    est2, rmsd2, cov2 = Rotation.align_vectors(a, b, 2 * cp.ones(5), True)

    assert_allclose(est1.as_matrix(), est2.as_matrix())
    assert_allclose(cp.sqrt(2) * rmsd1, rmsd2)
    assert_allclose(cov1, cov2)


def test_align_vectors_noise():
    rnd = cp.random.RandomState(0)
    n_vectors = 100
    rot = Rotation.random(random_state=rnd)
    vectors = rnd.normal(size=(n_vectors, 3))
    result = rot.apply(vectors)

    # The paper adds noise as independently distributed angular errors
    sigma = cp.deg2rad(1)
    tolerance = 1.5 * sigma
    noise = Rotation.from_rotvec(
        rnd.normal(
            size=(n_vectors, 3),
            scale=sigma
        )
    )

    # Attitude errors must preserve norm. Hence apply individual random
    # rotations to each vector.
    noisy_result = noise.apply(result)

    est, rmsd, cov = Rotation.align_vectors(noisy_result, vectors,
                                            return_sensitivity=True)

    # Use rotation compositions to find out closeness
    error_vector = (rot * est.inv()).as_rotvec()
    assert_allclose(error_vector[0], 0, atol=tolerance)
    assert_allclose(error_vector[1], 0, atol=tolerance)
    assert_allclose(error_vector[2], 0, atol=tolerance)

    # Check error bounds using covariance matrix
    cov *= sigma
    assert_allclose(cov[0, 0], 0, atol=tolerance)
    assert_allclose(cov[1, 1], 0, atol=tolerance)
    assert_allclose(cov[2, 2], 0, atol=tolerance)

    assert_allclose(rmsd, cp.sum((noisy_result - est.apply(vectors))**2)**0.5)


def test_align_vectors_single_vector():
    with pytest.warns(UserWarning, match="Optimal rotation is not"):
        r_estimate, rmsd = Rotation.align_vectors([[1, -1, 1]], [[1, 1, -1]])
        assert_allclose(rmsd, 0, atol=1e-16)


def test_align_vectors_invalid_input():
    with pytest.raises(ValueError, match="Expected input `a` to have shape"):
        Rotation.align_vectors([1, 2, 3], [[1, 2, 3]])

    with pytest.raises(ValueError, match="Expected input `b` to have shape"):
        Rotation.align_vectors([[1, 2, 3]], [1, 2, 3])

    with pytest.raises(ValueError, match="Expected inputs `a` and `b` "
                                         "to have same shapes"):
        Rotation.align_vectors([[1, 2, 3],[4, 5, 6]], [[1, 2, 3]])

    with pytest.raises(ValueError,
                       match="Expected `weights` to be 1 dimensional"):
        Rotation.align_vectors([[1, 2, 3]], [[1, 2, 3]], weights=[[1]])

    with pytest.raises(ValueError,
                       match="Expected `weights` to have number of values"):
        Rotation.align_vectors([[1, 2, 3]], [[1, 2, 3]], weights=[1, 2])

    with pytest.raises(ValueError,
                       match="`weights` may not contain negative values"):
        Rotation.align_vectors([[1, 2, 3]], [[1, 2, 3]], weights=[-1])


def test_random_rotation_shape():
    rnd = cp.random.RandomState(0)
    assert_equal(Rotation.random(random_state=rnd).as_quat().shape, (4,))
    assert_equal(Rotation.random(None, random_state=rnd).as_quat().shape, (4,))

    assert_equal(Rotation.random(1, random_state=rnd).as_quat().shape, (1, 4))
    assert_equal(Rotation.random(5, random_state=rnd).as_quat().shape, (5, 4))


def test_slerp():
    rnd = cp.random.RandomState(0)

    key_rots = Rotation.from_quat(rnd.uniform(size=(5, 4)))
    key_quats = key_rots.as_quat()

    key_times = [0, 1, 2, 3, 4]
    interpolator = Slerp(key_times, key_rots)

    times = [0, 0.5, 0.25, 1, 1.5, 2, 2.75, 3, 3.25, 3.60, 4]
    interp_rots = interpolator(times)
    interp_quats = interp_rots.as_quat()

    # Dot products are affected by sign of quaternions
    interp_quats[interp_quats[:, -1] < 0] *= -1
    # Checking for quaternion equality, perform same operation
    key_quats[key_quats[:, -1] < 0] *= -1

    # Equality at keyframes, including both endpoints
    assert_allclose(interp_quats[0], key_quats[0])
    assert_allclose(interp_quats[3], key_quats[1])
    assert_allclose(interp_quats[5], key_quats[2])
    assert_allclose(interp_quats[7], key_quats[3])
    assert_allclose(interp_quats[10], key_quats[4])

    # Constant angular velocity between keyframes. Check by equating
    # cos(theta) between quaternion pairs with equal time difference.
    cos_theta1 = cp.sum(interp_quats[0] * interp_quats[2])
    cos_theta2 = cp.sum(interp_quats[2] * interp_quats[1])
    assert_allclose(cos_theta1, cos_theta2)

    cos_theta4 = cp.sum(interp_quats[3] * interp_quats[4])
    cos_theta5 = cp.sum(interp_quats[4] * interp_quats[5])
    assert_allclose(cos_theta4, cos_theta5)

    # theta1: 0 -> 0.25, theta3 : 0.5 -> 1
    # Use double angle formula for double the time difference
    cos_theta3 = cp.sum(interp_quats[1] * interp_quats[3])
    assert_allclose(cos_theta3, 2 * (cos_theta1**2) - 1)

    # Miscellaneous checks
    assert_equal(len(interp_rots), len(times))


def test_slerp_rot_is_rotation():
    with pytest.raises(TypeError, match="must be a `Rotation` instance"):
        r = cp.array([[1,2,3,4],
                      [0,0,0,1]])
        t = cp.array([0, 1])
        Slerp(t, r)


def test_slerp_single_rot():
    msg = "must be a sequence of at least 2 rotations"
    with pytest.raises(ValueError, match=msg):
        r = Rotation.from_quat([1, 2, 3, 4])
        Slerp([1], r)


def test_slerp_rot_len1():
    msg = "must be a sequence of at least 2 rotations"
    with pytest.raises(ValueError, match=msg):
        r = Rotation.from_quat([[1, 2, 3, 4]])
        Slerp([1], r)


def test_slerp_time_dim_mismatch():
    with pytest.raises(ValueError,
                       match="times to be specified in a 1 dimensional array"):
        rnd = cp.random.RandomState(0)
        r = Rotation.from_quat(rnd.uniform(size=(2, 4)))
        t = cp.array([[1],
                      [2]])
        Slerp(t, r)


def test_slerp_num_rotations_mismatch():
    with pytest.raises(ValueError, match="number of rotations to be equal to "
                                         "number of timestamps"):
        rnd = cp.random.RandomState(0)
        r = Rotation.from_quat(rnd.uniform(size=(5, 4)))
        t = cp.arange(7)
        Slerp(t, r)


def test_slerp_equal_times():
    with pytest.raises(ValueError, match="strictly increasing order"):
        rnd = cp.random.RandomState(0)
        r = Rotation.from_quat(rnd.uniform(size=(5, 4)))
        t = [0, 1, 2, 2, 4]
        Slerp(t, r)


def test_slerp_decreasing_times():
    with pytest.raises(ValueError, match="strictly increasing order"):
        rnd = cp.random.RandomState(0)
        r = Rotation.from_quat(rnd.uniform(size=(5, 4)))
        t = [0, 1, 3, 2, 4]
        Slerp(t, r)


def test_slerp_call_time_dim_mismatch():
    rnd = cp.random.RandomState(0)
    r = Rotation.from_quat(rnd.uniform(size=(5, 4)))
    t = cp.arange(5)
    s = Slerp(t, r)

    with pytest.raises(ValueError,
                       match="`times` must be at most 1-dimensional."):
        interp_times = cp.array([[3.5],
                                 [4.2]])
        s(interp_times)


def test_slerp_call_time_out_of_range():
    rnd = cp.random.RandomState(0)
    r = Rotation.from_quat(rnd.uniform(size=(5, 4)))
    t = cp.arange(5) + 1
    s = Slerp(t, r)

    with pytest.raises(ValueError, match="times must be within the range"):
        s([0, 1, 2])
    with pytest.raises(ValueError, match="times must be within the range"):
        s([1, 2, 6])


def test_slerp_call_scalar_time():
    r = Rotation.from_euler('X', [0, 80], degrees=True)
    s = Slerp([0, 1], r)

    r_interpolated = s(0.25)
    r_interpolated_expected = Rotation.from_euler('X', 20, degrees=True)

    delta = r_interpolated * r_interpolated_expected.inv()

    assert_allclose(delta.magnitude(), 0, atol=1e-16)


def test_multiplication_stability():
    qs = Rotation.random(50, random_state=0)
    rs = Rotation.random(1000, random_state=1)
    for q in qs:
        rs *= q * rs
        assert_allclose(cp.linalg.norm(rs.as_quat(), axis=1), 1)


def test_pow():
    atol = 1e-14
    p = Rotation.random(10, random_state=0)
    p_inv = p.inv()
    # Test the short-cuts and other integers
    for n in [-5, -2, -1, 0, 1, 2, 5]:
        q = p ** n
        r = Rotation.identity(10)
        for _ in range(abs(n)):
            if n > 0:
                r = r * p
            else:
                r = r * p_inv
        ang = (q * r.inv()).magnitude()
        assert cp.all(ang < atol)

    # Large angle fractional
    for n in [-1.5, -0.5, -0.0, 0.0, 0.5, 1.5]:
        q = p ** n
        r = Rotation.from_rotvec(n * p.as_rotvec())
        assert_allclose(q.as_quat(), r.as_quat(), atol=atol)

    # Small angle
    p = Rotation.from_rotvec([1e-12, 0, 0])
    n = 3
    q = p ** n
    r = Rotation.from_rotvec(n * p.as_rotvec())
    assert_allclose(q.as_quat(), r.as_quat(), atol=atol)


def test_pow_errors():
    p = Rotation.random(random_state=0)
    with pytest.raises(NotImplementedError, match='modulus not supported'):
        pow(p, 1, 1)


def test_rotation_within_numpy_array():
    single = Rotation.random(random_state=0)
    multiple = Rotation.random(2, random_state=1)

    array = cp.array(single)
    assert_equal(array.shape, ())

    array = cp.array(multiple)
    assert_equal(array.shape, (2,))
    assert_allclose(array[0].as_matrix(), multiple[0].as_matrix())
    assert_allclose(array[1].as_matrix(), multiple[1].as_matrix())

    array = cp.array([single])
    assert_equal(array.shape, (1,))
    assert_equal(array[0], single)

    array = cp.array([multiple])
    assert_equal(array.shape, (1, 2))
    assert_allclose(array[0, 0].as_matrix(), multiple[0].as_matrix())
    assert_allclose(array[0, 1].as_matrix(), multiple[1].as_matrix())

    array = cp.array([single, multiple], dtype=object)
    assert_equal(array.shape, (2,))
    assert_equal(array[0], single)
    assert_equal(array[1], multiple)

    array = cp.array([multiple, multiple, multiple])
    assert_equal(array.shape, (3, 2))


def test_pickling():
    r = Rotation.from_quat([0, 0, cp.sin(np.pi/4), cp.cos(np.pi/4)])
    pkl = pickle.dumps(r)
    unpickled = pickle.loads(pkl)
    assert_allclose(r.as_matrix(), unpickled.as_matrix(), atol=1e-15)


def test_deepcopy():
    r = Rotation.from_quat([0, 0, cp.sin(np.pi/4), cp.cos(np.pi/4)])
    r1 = copy.deepcopy(r)
    assert_allclose(r.as_matrix(), r1.as_matrix(), atol=1e-15)


def test_as_euler_contiguous():
    r = Rotation.from_quat([0, 0, 0, 1])
    e1 = r.as_euler('xyz')  # extrinsic euler rotation
    e2 = r.as_euler('XYZ')  # intrinsic
    assert e1.flags['C_CONTIGUOUS'] is True
    assert e2.flags['C_CONTIGUOUS'] is True
    assert all(i >= 0 for i in e1.strides)
    assert all(i >= 0 for i in e2.strides)


def test_concatenate():
    rotation = Rotation.random(10, random_state=0)
    sizes = [1, 2, 3, 1, 3]
    starts = [0] + list(cp.cumsum(sizes))
    split = [rotation[i:i + n] for i, n in zip(starts, sizes)]
    result = Rotation.concatenate(split)
    assert_equal(rotation.as_quat(), result.as_quat())


def test_concatenate_wrong_type():
    with pytest.raises(TypeError, match='Rotation objects only'):
        Rotation.concatenate([Rotation.identity(), 1, None])


# Regression test for gh-16663
def test_len_and_bool():
    rotation_multi_empty = Rotation(cp.empty((0, 4)))
    rotation_multi_one = Rotation([[0, 0, 0, 1]])
    rotation_multi = Rotation([[0, 0, 0, 1], [0, 0, 0, 1]])
    rotation_single = Rotation([0, 0, 0, 1])

    assert len(rotation_multi_empty) == 0
    assert len(rotation_multi_one) == 1
    assert len(rotation_multi) == 2
    with pytest.raises(TypeError, match="Single rotation has no len()."):
        len(rotation_single)

    # Rotation should always be truthy. See gh-16663
    assert rotation_multi_empty
    assert rotation_multi_one
    assert rotation_multi
    assert rotation_single

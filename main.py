import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32, fast_math=True)

# Poisson over grid?
# Small scales use direct comparison??

# Hamiltonian go brrrr :)

# |x|x|x|x|x|
# |x|x|x|x|x|
# |x|x|x|x|x|
# |x|x|x|x|x|
# |x|x|x|x|x|

# If each of n boxes has width dx, total width is dx * n, duh
# But if we want the grid to have each box centered at x, the corners are a x - dx/2, x + dx/2
# Obviously, this means that the center of the grid, x, at index i is at x_min + dx * (i + 0.5)
# To calculate i given x, we can just invert, giving us floor((x - x_min) / dx)

particle_count = 160000
# dispersion_radius = 3  # Spread mass with a truncated gaussian distribution over this radius
# scaling = 5  # Calculate the perturbations induced by particles within a manhattan distance (|∆x| + |∆y|) of 5

G, c, rho = 100, 1, 10
pi = 3.141592


(x_min, x_max), (y_min, y_max) = x_bounds, y_bounds = bounds = (-10, 10), (-10, 10)
x_range, y_range = x_max - x_min, y_max - y_min
x_res, y_res = resolution = (3840, 2160)

dx = (x_max - x_min) / x_res
dy = (y_max - y_min) / y_res

dxs = ti.Vector([dx, dy])
scale = 5 * rho

pos, vel = ti.Vector.field(2, ti.f32), ti.Vector.field(2, ti.f32)

density = ti.field(ti.f32)
energy = ti.field(ti.f32)
pixels = ti.Vector.field(3, ti.f32, shape=(1920, 1080))


ti.root.bitmasked(ti.i, (particle_count,)).place(pos, vel)

# ti.root.pointer(ti.ij, (x_res // 16, y_res // 16)).dense(ti.ij, (16, 16)).place(
#     energy, density
# )  # Improve cache-hit probability, while maximizing sparsity

ti.root.dense(ti.ij, (x_res // 16, y_res // 16)).dense(ti.ij, (16, 16)).place(
    energy, density
)  # Improve cache-hit probability

gui = ti.GUI(res=pixels.shape, fast_gui=True)
sx, sy = x_res // pixels.shape[0], y_res // pixels.shape[1]


@ti.func
def get_position(i: ti.u32, j: ti.u32) -> ti.Vector:
    return ti.Vector([x_min + (i + 0.5) * dx, y_min + (j + 0.5) * dy])


@ti.func
def get_index(x: ti.f32, y: ti.f32) -> ti.Vector:
    return ti.cast(
        ti.floor(ti.Vector([(x - x_min) / dx, (y - y_min) / dy]) - 0.5), ti.int32
    )  # Might have to move the floor to each argument lol


@ti.func
def get_index_f(x: ti.f32, y: ti.f32) -> ti.Vector:
    return ti.Vector([(x - x_min) / dx, (y - y_min) / dy]) - 0.5


@ti.kernel
def initialize():
    for i in range(particle_count):
        r, theta = 2*(ti.random() ** 0.5), 2 * pi * ti.random()
        if ti.random() > 0.5:
            pos[i] = ti.Vector(
                [-5.0 + r * ti.cos(theta), 0.0 + r * ti.sin(theta)]
            )  # Random Position (within range, of course)
            vel[i] = ti.Vector([1000, 0])
        else:
            pos[i] = ti.Vector(
                [5.0 + r * ti.cos(theta), 0.0 + r * ti.sin(theta)]
            )  # Random Position (within range, of course)
            vel[i] = ti.Vector([-1000, 0])
        # vel[i] = 0.0 * ti.Vector([ti.randn(), ti.randn()])  # Random Velocity


@ti.func
def is_valid(i: ti.i32, j: ti.i32) -> ti.i8:
    idxs = ti.Vector([i, j], ti.i32)
    return not (
        (idxs > ti.Vector([x_res - 1, y_res - 1])).any()
        or (idxs < ti.Vector([0, 0])).any()
    )


# @ti.func
# def shape_scalar(v, idx: ti.i32, idy: ti.i32, field):
#     coefficients = ti.field(ti.f32, shape=(dispersion_radius, dispersion_radius))

#     for I in ti.grouped(coefficients):
#         if (I).norm()


@ti.kernel
def set_density():
    ti.block_dim(256)
    for i in ti.grouped(density):
        density[i] = 0
    
    for i in pos:
        fidx = get_index_f(*pos[i])
        idx = ti.cast(ti.floor(fidx), ti.int32)
        rem = ti.abs(fidx - idx)
        # density[idx.x, idx.y] += rho

        if is_valid(idx.x, idx.y):
            density[idx.x, idx.y] += (1 - rem.x) * (1 - rem.y) * rho

        if is_valid(idx.x + 1, idx.y):
            density[idx.x + 1, idx.y] += rem.x * (1 - rem.y) * rho

        if is_valid(idx.x, idx.y + 1):
            density[idx.x, idx.y + 1] += (1 - rem.x) * rem.y * rho

        if is_valid(idx.x + 1, idx.y + 1):
            density[idx.x + 1, idx.y + 1] += rem.x * rem.y * rho

        # Use a shape function to describe how to the particle masses are transferred to the grid
        # Should I deactivate all unused cells??? (I mean, probably lmao, but also.... I'm lazy)
        # Shape function should take a index, value, and the array(????)


@ti.kernel
def refine_energy():
    ti.block_dim(256)
    for I in ti.grouped(energy):
        idxs = [
            I + ti.Vector.unit(2, 0),
            I - ti.Vector.unit(2, 0),
            I + ti.Vector.unit(2, 1),
            I - ti.Vector.unit(2, 1),
        ]
        energies = [
            energy[idxs[0]] if is_valid(*idxs[0]) else 0,
            energy[idxs[1]] if is_valid(*idxs[1]) else 0,
            energy[idxs[2]] if is_valid(*idxs[2]) else 0,
            energy[idxs[3]] if is_valid(*idxs[3]) else 0,
        ]
        energy[I] = (
            0.5
            * (
                dy ** 2 * (energies[0] + energies[1])
                + dx ** 2 * (energies[2] + energies[3])
                - (dx ** 2) * (dy ** 2) * (4 * pi * G * density[I])
            )
            / (dx ** 2 + dy ** 2)
        )


@ti.func
def gradient(field, idx):
    return ti.Vector(
        [
            (field[idx + ti.Vector.unit(2, 0)] - field[idx - ti.Vector.unit(2, 0)])
            / (2 * dxs.x),
            (field[idx + ti.Vector.unit(2, 1)] - field[idx - ti.Vector.unit(2, 1)])
            / (2 * dxs.y),
        ]
    )


@ti.kernel
def update(dt: ti.f32):
    for i in pos:
        fidx = get_index_f(*pos[i])
        idx = ti.cast(ti.floor(fidx), ti.int32)
        rem = ti.abs(fidx - idx)

        vel[i] += (
            -dt
            * (
                gradient(energy, idx + ti.Vector([0, 0])) * (1 - rem.x) * (1 - rem.y)
                + gradient(energy, idx + ti.Vector([1, 0])) * rem.x * (1 - rem.y)
                + gradient(energy, idx + ti.Vector([0, 1])) * (1 - rem.x) * rem.y
                + gradient(energy, idx + ti.Vector([1, 1])) * rem.x * rem.y
            )
            / (
                (
                    density[idx + ti.Vector([0, 0])] * (1 - rem.x) * (1 - rem.y)
                    + density[idx + ti.Vector([1, 0])] * rem.x * (1 - rem.y)
                    + density[idx + ti.Vector([0, 1])] * (1 - rem.x) * rem.y
                    + density[idx + ti.Vector([1, 1])] * rem.x * rem.y
                )
                * dx
                * dy
            )
        )

        # vel[i] *= 0.999
        pos[i] += vel[i] * dt

        if (pos[i] > ti.Vector([x_max, y_max])).any() or (
            pos[i] < ti.Vector([x_min, y_min])
        ).any():
            # pos[i] = (
            #     (pos[i] - ti.Vector([x_min, y_min]))
            #     % ti.Vector([x_max - x_min, y_max - y_min])
            # ) + ti.Vector([x_min, y_min])
            ti.deactivate(pos.snode.parent(), [i])


@ti.kernel
def set_pixels():
    for i in ti.grouped(pixels):
        pixels[i] = ti.Vector([0, 0, 0])
        for u in ti.static(range(sx)):
            for v in ti.static(range(sy)):
                pixels[i.x, i.y] += density[sx * i.x + u, sy * i.y + v] / scale


if __name__ == "__main__":
    # video_manager = ti.VideoManager(
    #     output_dir="./results", framerate=60, automatic_build=False
    # )
    gui = ti.GUI(res=pixels.shape)
    gui.fps_limit = 10

    from time import time

    # t = time()
    initialize()
    while gui.running:
    # for i in range(100 * 60):
        set_pixels()

        for n in range(10):
            set_density()

            for m in range(10):
                refine_energy()

            # temp_t = time()
            # dt = temp_t - t
            # t = temp_t
            update(1e-5)

        # video_manager.write_frame(pixels.to_numpy())
        # print(f"\rFrame {i+1}/{100 * 60} is recorded", end="")
        gui.set_image(pixels)
        gui.show()

        # break
    
    # ti.kernel_profiler_print()
    # ti.print_profile_info()
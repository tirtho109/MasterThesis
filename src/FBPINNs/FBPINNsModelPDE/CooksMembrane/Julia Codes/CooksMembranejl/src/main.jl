#=
    This code is for generating training data also for result comparison of FBPINNs-
    and FEM results.
    Author: Tirtho Sarathi Saha
=#
include("FeFunctions.jl")

λm = 4                           #units are in Pa
μm = 5
mp = LinearElasticity(λm, μm)

nx = 100
ny = 100

grid = create_mesh(nx, ny)


# DofHandler and ConstraintHandler
dimension = 2;
reference_shape = RefTetrahedron #in 2D it's Triangle
polynomial_order = 1 #linear approximation

interpolation = Lagrange{dimension, reference_shape, polynomial_order}()
dh = create_dofhandler(grid, interpolation)

quadrature_order = 1
qr = QuadratureRule{dimension, reference_shape}(quadrature_order)
qr_face = QuadratureRule{dimension-1, RefTetrahedron}(quadrature_order)
cv = CellVectorValues(qr, interpolation)
fv = FaceVectorValues(qr_face, interpolation)

xs = [
    Vec{2}((1.2, 4.5)),
    Vec{2}((4.5, 5.5)),
    Vec{2}((2.5, 7.1))
]
#Ke, fe = element_assembly!(Ke, fe, xs, cv, fv, mp, ΓN)

#=
# Generate DofHandler
dh = DofHandler(grid)
dim = 2
push!(dh, :u, dim)
close!(dh)
e = 50
edof = celldofs(dh,e)
=#

# Boundary Condition
# Dirichlet BC
ch = ConstraintHandler(dh)
dbc_left = Dirichlet(
    :u,
    getfaceset(grid, "left"),
    (x,t) -> [0,0],
    [1,2]
)

add!(ch, dbc_left)
close!(ch)
update!(ch, 0.0)

# Neumann Part of the Boundary

ΓN = getfaceset(grid, "right")

K = create_sparsity_pattern(dh)

K, f = doassemble(K,cv,fv,grid, dh, mp, ΓN)

apply!(K, f, ch)

u = K \ f

apply!(u, ch)

# postprocessing
σ_vm = calculate_stress(grid, dh, cv, u, mp)

vtk_grid("assignment$(nx)_$(ny)", dh) do vtk
    # Export the solution using the DofHandler dh and solution u
    vtk_point_data(vtk, dh, u)
    # Export the von von_misses_stress
    vtk_cell_data(vtk, σ_vm, "von_misses_stress" )
end

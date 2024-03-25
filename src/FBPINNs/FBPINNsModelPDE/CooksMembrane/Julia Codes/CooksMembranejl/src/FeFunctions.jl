using Ferrite, Tensors, SparseArrays

# The material is linear elastic, which is here specified by the shear and bulk moduli
struct LinearElasticity
    λ::Float64
    μ::Float64
end

# Creating FE grid/mesh
# create_mesh(element_in_x, element_in_y)...return grid
function create_mesh(nx::Int, ny::Int)
    corners = [
                Vec{2}((0.0, 0.0)),
                Vec{2}((0.048, 0.044)),
                Vec{2}((0.048, 0.044+0.016)),
                Vec{2}((0.0, 0.044))
    ]
    grid = generate_grid(Triangle, (nx,ny), corners);
    return grid
end
#=
grid = create_mesh(10,10);
element = 50;
coord = getcoordinates(grid,element)
=#

# Function for creating cell values (see Exercise-2)
function create_values(interpolation)
    # quadrature rule
    qr = QuadratureRule{2,RefTetrahedron}(1)
    # cellvalues
    cellvalues = CellVectorValues(qr, interpolation)
    return cellvalues
end

# Function for creating a DofHandler, which distributes and takes care of all the
# degrees-of-freedom (dofs) in the problem.
function create_dofhandler(grid, interpolation)
    dh = DofHandler(grid)
    # Add a field :u with two components (x- and y-displacement)
    push!(dh, :u, 2, interpolation)
    # Finalize the dofhandler
    close!(dh)
    return dh
end



# Task A

# 4th-O Elasticity Tensor
function calculate_E(x)
    λ = x.λ
    μ = x.μ
    δ(i,j) = i == j ? 1.0 : 0.0
    EE = SymmetricTensor{4, 2}(
        (i,j,k,l) -> λ * δ(i,j) * δ(k,l) + μ * (δ(i,k) * δ(j,l) + δ(i,l) * δ(j,k))
    )    
    return EE;
end

# Element stiffness matrix and force vector
function element_assembly!(ke::Matrix, fe::Vector, cell, cellvalues::CellVectorValues, facevalues::FaceVectorValues, x::LinearElasticity, ΓN, dh::DofHandler)

    E = calculate_E(x)
    # 1. Reset ke (might have values from previous element)
    fill!(ke,0)
    fill!(fe,0)
    be = zeros(ndofs_per_cell(dh))
    # 2. Re-initialized cached values for this element
    reinit!(cellvalues, cell)
    # 3. Loop over the quadrature points
    #calculate Ke
    for qp in 1: getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, qp)
        # 4. Loop over the shape functions for the test function
        for i in 1: getnbasefunctions(cellvalues)
            ϵi = shape_symmetric_gradient(cellvalues, qp, i)
            # 5. Loop over the shape functions for the trial function
            for j in 1: getnbasefunctions(cellvalues)
                ϵj = shape_symmetric_gradient(cellvalues, qp, j)
                # 6. Compute contribution ke[i, j]
                ke[i,j] += ϵi ⊡ E ⊡ ϵj * dΩ
            end
            b = (0.0 , 0.0)
            δui = shape_value(cellvalues, qp, i)
            be[i] += δui ⋅ b * dΩ
        end 
    end
    t =  Vec{2}((0.0, 1)) # Traction (to be scaled with surface normal) #1sOT using this
    # Calculate traction in the right side
    for face in 1:nfaces(cell)
        if(cellid(cell), face) in ΓN
            reinit!(facevalues, cell, face)
            for q_point in 1:getnquadpoints(facevalues)
                #t = tn * getnormal(facevalues, q_point)
                dΓ = getdetJdV(facevalues, q_point)
                for i in 1:getnbasefunctions(cellvalues)
                    δui = shape_value(facevalues, q_point, i)
                    fe[i] += (δui ⋅ t) * dΓ # check -/+
                end
            end
        end
    end
    fe .+= be
    return ke, fe
end


function doassemble(K::SparseMatrixCSC,cellvalues::CellVectorValues, facevalues::FaceVectorValues, grid, dh::DofHandler, x::LinearElasticity, ΓN)
    ndpc = ndofs_per_cell(dh)
    #elementwise
    Ke = zeros(ndpc, ndpc)
    fe = zeros(ndpc)
    #Global
    #K = zeros(ndofs(dh), ndofs(dh))
    f = zeros(ndofs(dh))
    assembler = start_assemble(K,f)
    for cell in CellIterator(dh)
        element_assembly!(Ke, fe, cell, cellvalues, facevalues, x, ΓN, dh)
        assemble!(assembler, celldofs(cell), Ke, fe)
    end

    return K, f
end


# postprocessing
function calculate_stress(grid, dh, cellvalues, u, mp)

    # Calculate E
    E = calculate_E(mp)
    # Vector to collect output -- one value for every element
    stress = zeros(getncells(grid))

    # Loop over the elelments
    for ei in 1:getncells(grid)
        # Fetch coordinates for this cell (need to upadate cellvalues)
        coords = getcoordinates(grid, ei)

        # Reinit cellbalues
        reinit!(cellvalues, coords)

        # Fetch the dofs for this cell
        dofs = celldofs(dh, ei)

        # Part of the solution belonging to this element
        ue = u[dofs]

        # #of quadrature ppint
        qp = 1

        # Compute strain from the (local) solution  ae
        ϵ = function_symmetric_gradient(cellvalues, qp, ue)

        # Compute stress
        σ = E ⊡ ϵ

        # Compute von_misses_stress
        σ_vm = sqrt(3/2 * dev(σ) ⊡ dev(σ))

        # Add to our output vector
        stress[ei] = σ_vm
    end
    # return
    return stress
end


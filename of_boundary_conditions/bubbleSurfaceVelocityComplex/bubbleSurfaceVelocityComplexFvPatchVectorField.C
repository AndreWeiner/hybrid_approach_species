/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2004-2010 OpenCFD Ltd.
     \\/     M anipulation  |
-------------------------------------------------------------------------------
                            | Copyright (C) 2011-2016 OpenFOAM Foundation
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "bubbleSurfaceVelocityComplexFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "surfaceFields.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::bubbleSurfaceVelocityComplexFvPatchVectorField::
bubbleSurfaceVelocityComplexFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(p, iF),
    origin_(Zero),
    axis_(Zero),
    normal_(Zero),
    model_name_velocity_("")
{}


Foam::bubbleSurfaceVelocityComplexFvPatchVectorField::
bubbleSurfaceVelocityComplexFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchField<vector>(p, iF, dict, false),
    origin_(dict.lookup("origin")),
    axis_(dict.lookup("axis")),
    normal_(dict.lookup("normal")),
    model_name_velocity_(dict.lookupOrDefault<word>("velocity_model", "velocity_model.pt"))
{
    pyTorch_velocity_ = torch::jit::load(model_name_velocity_);
    if (!dict.found("value"))
    {
      updateCoeffs();
    }
}


Foam::bubbleSurfaceVelocityComplexFvPatchVectorField::
bubbleSurfaceVelocityComplexFvPatchVectorField
(
    const bubbleSurfaceVelocityComplexFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchField<vector>(ptf, p, iF, mapper),
    origin_(ptf.origin_),
    axis_(ptf.axis_),
    normal_(ptf.normal_),
    model_name_velocity_(ptf.model_name_velocity_),
    pyTorch_velocity_(ptf.pyTorch_velocity_)
{}


Foam::bubbleSurfaceVelocityComplexFvPatchVectorField::
bubbleSurfaceVelocityComplexFvPatchVectorField
(
    const bubbleSurfaceVelocityComplexFvPatchVectorField& rwvpvf
)
:
    fixedValueFvPatchField<vector>(rwvpvf),
    origin_(rwvpvf.origin_),
    axis_(rwvpvf.axis_),
    normal_(rwvpvf.normal_),
    model_name_velocity_(rwvpvf.model_name_velocity_),
    pyTorch_velocity_(rwvpvf.pyTorch_velocity_)
{}


Foam::bubbleSurfaceVelocityComplexFvPatchVectorField::
bubbleSurfaceVelocityComplexFvPatchVectorField
(
    const bubbleSurfaceVelocityComplexFvPatchVectorField& rwvpvf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(rwvpvf, iF),
    origin_(rwvpvf.origin_),
    axis_(rwvpvf.axis_),
    normal_(rwvpvf.normal_),
    model_name_velocity_(rwvpvf.model_name_velocity_),
    pyTorch_velocity_(rwvpvf.pyTorch_velocity_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::bubbleSurfaceVelocityComplexFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    // compute tangent vector
    const vectorField n(patch().nf());
    const vectorField tau(n ^ normal_);

    // face center position vectors
    const vectorField Cf(patch().Cf() - origin_);

    torch::Tensor xyTensor = torch::ones({Cf.size(), 2}, torch::kFloat64);

    forAll(Cf, faceI)
    {
        xyTensor[faceI][0] = Cf[faceI] & (normal_ ^ axis_);
	xyTensor[faceI][1] = Cf[faceI] & axis_;
    }

    // run forward pass to compute tangential velocity
    std::vector<torch::jit::IValue> modelFeatures{xyTensor};
    torch::Tensor uTensor = pyTorch_velocity_.forward(modelFeatures).toTensor();
    auto uAccessor = uTensor.accessor<double,1>();

    vectorField surfaceVelocity(Cf.size(), Zero);
    forAll(surfaceVelocity, faceI)
    {
	surfaceVelocity[faceI] = tau[faceI] * uAccessor[faceI];
    }

    vectorField::operator=(surfaceVelocity);
    fixedValueFvPatchVectorField::updateCoeffs();
}


void Foam::bubbleSurfaceVelocityComplexFvPatchVectorField::write(Ostream& os) const
{
    fvPatchVectorField::write(os);
    os.writeEntry("origin", origin_);
    os.writeEntry("axis", axis_);
    os.writeEntry("normal", normal_);
    os.writeEntry("velocity_model", model_name_velocity_);
    writeEntry("value", os);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField
    (
        fvPatchVectorField,
        bubbleSurfaceVelocityComplexFvPatchVectorField
    );
}

// ************************************************************************* //

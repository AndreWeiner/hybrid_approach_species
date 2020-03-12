#include "fvCFD.H"
#include "mathematicalConstants.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    instantList timeDirs = timeSelector::select0(runTime, args);
    // determine patch ID
    label surfaceID(-1);
    forAll (mesh.boundary(), patchI)
    {
        if (mesh.boundary()[patchI].name() == "bubble")
        {
            surfaceID = patchI;
        }
    }

    const vectorField Cf(mesh.Cf().boundaryField()[surfaceID]);
    const vectorField Sf(mesh.Sf().boundaryField()[surfaceID]);

    scalarField globalGrad(timeDirs.size(), 0.0);

    forAll (timeDirs, timei)
    {
        runTime.setTime(timeDirs[timei], timei);
        Info<< "Time = " << runTime.timeName() << endl;

        #include "createFields.H"

        scalarField snGradT(T.boundaryField()[surfaceID].snGrad());
        globalGrad[timei] = sum(snGradT * mag(Sf)) / sum(mag(Sf));

        OFstream outputFile(runTime.path()/runTime.timeName()/"snGradT.csv");
        outputFile.precision(15);
        outputFile  << "# x, y, A, snGradT";

        forAll(Cf, faceI)
        {
            outputFile << "\n"
                      << Cf[faceI].x() << ", " << Cf[faceI].y() << ", "
                      << mag(Sf[faceI]) << ", " << snGradT[faceI];
    
        }
    } // end of time loop

    // write global gradient
    OFstream gradFile(runTime.path()/"snGradTGlobal.csv");
    gradFile.precision(15);
    gradFile << "# time, snGradT, A";

    forAll (timeDirs, timei)
    {
        runTime.setTime(timeDirs[timei], timei);
        gradFile << "\n"
                << runTime.timeName() << ", "
                << globalGrad[timei] << ", "
                << sum(mag(Sf));
    }

    Info << "End\n" << endl;

    return 0;
}

// ************************************************************************* //

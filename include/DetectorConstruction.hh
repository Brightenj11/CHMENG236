//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
/// \file DetectorConstruction.hh
/// \brief Definition of the B1::DetectorConstruction class

#ifndef B1DetectorConstruction_h
#define B1DetectorConstruction_h 1

#include "G4UImessenger.hh"
#include "G4VUserDetectorConstruction.hh"
#include "globals.hh"

#include <memory>
#include <vector>

class G4UIcmdWithAString;
class G4VPhysicalVolume;
class G4LogicalVolume;

namespace B1
{

/// 1 cm cubes on a 9 x 9 x 9 lattice, set with /geom/addCube in a macro before /run/initialize,
/// inside a fixed spherical detector.

class DetectorConstruction : public G4VUserDetectorConstruction, public G4UImessenger
{
  public:
    DetectorConstruction();
    ~DetectorConstruction() override;

    G4VPhysicalVolume* Construct() override;
    void SetNewValue(G4UIcommand* command, G4String value) override;

    G4LogicalVolume* GetScoringVolume() const { return fScoringVolume; }

  protected:
    G4LogicalVolume* fScoringVolume = nullptr;

  private:
    struct Cube
    {
        G4int i = 0, j = 0, k = 0;
        G4String material;
    };

    std::unique_ptr<G4UIcmdWithAString> fAddCubeCmd;
    std::vector<Cube> fCubes;
};

}  // namespace B1

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif

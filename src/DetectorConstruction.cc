//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The Geant4 software is copyright of the Copyright Holders of     *
// * the Geant4 Collaboration.  It is provided under the terms and    *
// * conditions of the Geant4 Software License, included in the file *
// * LICENSE and available at http://cern.ch/geant4/license .  These  *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes, nor the agencies providing financial support for this *
// * work  make  any representation or warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using, copying, modifying or distributing the software (or    *
// * any work based on the software)  you agree to acknowledge its    *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
/// \file DetectorConstruction.cc
/// \brief Implementation of the B1::DetectorConstruction class

#include "DetectorConstruction.hh"

#include "G4Box.hh"
#include "G4Exception.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4Sphere.hh"
#include "G4SystemOfUnits.hh"
#include "G4UIcmdWithAString.hh"

#include <sstream>

namespace B1
{

namespace
{
constexpr G4int kLatticeSize = 9;  // positions per side. Odd, so a cube can be centered on the beam
constexpr G4double kCubeSide = 1. * cm;
}  // namespace

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::DetectorConstruction()
  : fAddCubeCmd(std::make_unique<G4UIcmdWithAString>("/geom/addCube", this))
{
  fAddCubeCmd->SetGuidance("Place a 1 cm cube: i j k material. i, j, k run from 0 to 8.");
  fAddCubeCmd->SetGuidance("Example: /geom/addCube 4 4 4 G4_Al. Position 4 4 4 is the center.");
  fAddCubeCmd->SetParameterName("cube", false);
  fAddCubeCmd->AvailableForStates(G4State_PreInit);  // the geometry is built at /run/initialize
  fAddCubeCmd->SetToBeBroadcasted(false);  // this object only exists on the master thread
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::~DetectorConstruction() = default;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorConstruction::SetNewValue(G4UIcommand*, G4String value)
{
  Cube cube;
  std::istringstream fields(value);
  G4String extra;
  const G4bool parsed = static_cast<bool>(fields >> cube.i >> cube.j >> cube.k >> cube.material)
                        && !(fields >> extra);
  const auto inLattice = [](G4int index) { return index >= 0 && index < kLatticeSize; };
  if (!parsed || !inLattice(cube.i) || !inLattice(cube.j) || !inLattice(cube.k)) {
    G4Exception("DetectorConstruction::SetNewValue", "Geom001", FatalException,
                ("/geom/addCube needs i j k material with i, j, k from 0 to 8. Got: " + value)
                  .c_str());
  }
  for (const auto& other : fCubes) {
    if (other.i == cube.i && other.j == cube.j && other.k == cube.k) {
      G4Exception("DetectorConstruction::SetNewValue", "Geom002", FatalException,
                  ("Lattice position already has a cube: " + value).c_str());
    }
  }
  fCubes.push_back(cube);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  G4NistManager* nist = G4NistManager::Instance();
  G4bool checkOverlaps = true;

  // Vacuum, so everything recorded at the detector comes from the cubes and not from air
  G4Material* vacuum = nist->FindOrBuildMaterial("G4_Galactic");

  //
  // World
  //
  auto solidWorld = new G4Box("World", 1.4 * m, 1.4 * m, 1.4 * m);
  auto logicWorld = new G4LogicalVolume(solidWorld, vacuum, "World");
  auto physWorld = new G4PVPlacement(nullptr, G4ThreeVector(), logicWorld, "World", nullptr,
                                     false, 0, checkOverlaps);

  //
  // Cubes on a 9 x 9 x 9 lattice centered at the origin. Position (4, 4, *) is on the beam axis.
  //
  auto solidCube = new G4Box("Cube", 0.5 * kCubeSide, 0.5 * kCubeSide, 0.5 * kCubeSide);
  const G4int center = kLatticeSize / 2;
  for (const auto& cube : fCubes) {
    G4Material* material = nist->FindOrBuildMaterial(cube.material);
    if (!material) {
      G4Exception("DetectorConstruction::Construct", "Geom003", FatalException,
                  ("Unknown NIST material: " + cube.material).c_str());
    }
    auto logicCube = new G4LogicalVolume(solidCube, material, "Cube");
    const G4ThreeVector position((cube.i - center) * kCubeSide, (cube.j - center) * kCubeSide,
                                 (cube.k - center) * kCubeSide);
    new G4PVPlacement(nullptr, position, logicCube, "Cube", logicWorld, false, 0, checkOverlaps);
  }

  //
  // Detector: full vacuum sphere centered on the lattice and fixed for every geometry, so every
  // outgoing direction (theta = 0 to 180 deg) is covered. The gun at z = -50 cm is inside it, so
  // incoming protons never cross the detector on their way to the cubes. SteppingAction records
  // particles and kills them on entry, so it acts as an ideal counter and never perturbs what it
  // measures.
  //
  G4double detRadius = 1.3 * m;
  auto solidDet = new G4Sphere("Detector", detRadius, detRadius + 1. * mm, 0., 360. * deg, 0.,
                               180. * deg);
  auto logicDet = new G4LogicalVolume(solidDet, vacuum, "Detector");
  new G4PVPlacement(nullptr, G4ThreeVector(), logicDet, "Detector", logicWorld, false, 0,
                    checkOverlaps);

  fScoringVolume = logicDet;

  return physWorld;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

}  // namespace B1

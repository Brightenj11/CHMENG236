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
/// \file SteppingAction.cc
/// \brief Implementation of the B1::SteppingAction class

#include "SteppingAction.hh"

#include "DetectorConstruction.hh"

#include "G4AnalysisManager.hh"
#include "G4Event.hh"
#include "G4LogicalVolume.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"

namespace B1
{

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void SteppingAction::UserSteppingAction(const G4Step* step)
{
  if (!fScoringVolume) {
    const auto detConstruction = static_cast<const DetectorConstruction*>(
      G4RunManager::GetRunManager()->GetUserDetectorConstruction());
    fScoringVolume = detConstruction->GetScoringVolume();
  }

  // Post-step volume is null when a particle leaves the world
  const G4StepPoint* post = step->GetPostStepPoint();
  const G4VPhysicalVolume* postVolume = post->GetPhysicalVolume();
  if (!postVolume || postVolume->GetLogicalVolume() != fScoringVolume) return;

  // Entering the detector: every track stops here, so each is counted exactly once
  step->GetTrack()->SetTrackStatus(fStopAndKill);

  const G4String& name = step->GetTrack()->GetDefinition()->GetParticleName();
  if (name != "proton" && name != "neutron") return;

  const G4ThreeVector dir = post->GetMomentumDirection();
  auto analysis = G4AnalysisManager::Instance();
  analysis->FillNtupleIColumn(0, G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID());
  analysis->FillNtupleSColumn(1, name);
  analysis->FillNtupleDColumn(2, post->GetKineticEnergy() / MeV);
  analysis->FillNtupleDColumn(3, dir.theta() / deg);  // angle from the beam axis (+z)
  analysis->FillNtupleDColumn(4, dir.phi() / deg);
  analysis->AddNtupleRow();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

}  // namespace B1

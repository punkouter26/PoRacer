# Publishing PoRacer

One command builds the signed AAB and uploads it to Play **internal testing** as a **draft**:

    .\Scripts\publish.ps1

- `-DryRun` rehearses everything and lands nothing in the console.
- `-SkipBuild` uploads the existing `Builds\Android$app.aab`.
- Close this project in the Unity editor first, or the headless build fails on the lock.
- bundleVersionCode is auto-bumped before each build (pass `-NoBump` to keep it);
  Play rejects duplicate versionCodes, so the bump is the default.
- Store listing text/artwork lives in `C:\Users\punko\Downloads\PlayStoreUploads`
  (`play_listing.py`), the AAB uploader is this repo's `Tools\play_publish.py`, and both
  authenticate with the service-account key at
  `C:\Users\punko\Downloads\PoRacer-Release\play-service-account.json`.
- Rolling a draft out to testers, content rating, and data safety are manual Play Console steps.

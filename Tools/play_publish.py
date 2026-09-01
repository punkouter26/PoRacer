#!/usr/bin/env python3
"""Upload a signed AAB to the Google Play Console via the Play Developer API v3.

WHY THIS EXISTS. There is no way to reach the Play Console from a headless
session: it is a browser product behind an interactive Google login. The
Developer API is the supported non-interactive route, and it authenticates with
a SERVICE ACCOUNT key rather than a human account -- which is also why this
script can never act "as punkouter27". It acts as a robot account that the human
account has granted release permission to. That grant is the one-time manual
step no script can do for you; see SETUP below.

SAFETY MODEL, because this is the one tool here that talks to the outside world:

  - The default track is `internal`, NOT production. Shipping to users is opt-in.
  - The default release status is `draft`. A draft is uploaded and visible in the
    console but is NOT rolled out to anybody until a human presses the button.
    Getting `completed` requires typing it.
  - `--dry-run` does the whole flow -- edit, upload, track write -- and then
    DELETES the edit instead of committing. Nothing an uncommitted edit does is
    visible to users, so this is a genuine rehearsal and not an approximation.
  - The current state of the target track is printed BEFORE the upload, so a
    versionCode collision is something you see rather than something you find
    out from an API error.

SETUP (one time, and it needs a human in a browser):

  1. Play Console -> Create app -> PoRacer. The application id below has to be
     the one the app is created with; it is permanent.
  2. Google Cloud console -> the project linked to your Play account ->
     IAM & Admin -> Service Accounts -> Create. No GCP roles are needed.
  3. On that service account, Keys -> Add key -> JSON. Download it.
  4. Play Console -> Users and permissions -> Invite new user -> paste the
     service account email -> grant "Release to production, exclude devices,
     and use Play App Signing" (or narrower) for the PoRacer app.
     The invite must be for the APP, and permission changes take a few minutes
     to propagate -- a 401 right after granting is usually just impatience.
  5. Put the JSON next to the keystore, which is already the project's
     convention for secrets that must stay out of the repo:
       C:/Users/punko/Downloads/PoRacer-Release/play-service-account.json

  The FIRST bundle for a brand new app cannot be uploaded through this script:
  Play will not accept an API upload until the app's store listing, content
  rating and data-safety forms are filled in, and the internal testing track has
  a tester list. Do that first upload by hand in the console, then this script
  owns every one after it.

USAGE

  python Tools/play_publish.py --dry-run
  python Tools/play_publish.py --track internal --status completed
  python Tools/play_publish.py --track production --status draft
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The application id is permanent once published. Editor_BuildAndroidAAB.APP_ID is
# the one that actually ships; this must match it or the API 404s on an app that
# does not exist under this developer account.
PACKAGE_NAME = "com.punkoutersoftware.poracer"

DEFAULT_AAB = os.path.join(REPO_ROOT, "Builds", "Android", "PoRacer.aab")
DEFAULT_CREDENTIALS = "C:/Users/punko/Downloads/PoRacer-Release/play-service-account.json"

API_ROOT = "https://androidpublisher.googleapis.com/androidpublisher/v3"
UPLOAD_ROOT = "https://androidpublisher.googleapis.com/upload/androidpublisher/v3"
SCOPE = "https://www.googleapis.com/auth/androidpublisher"


def fail(message):
    print("PLAY PUBLISH RESULT: Aborted — " + message, file=sys.stderr)
    sys.exit(1)


def access_token(credentials_path):
    """Exchange the service account key for an OAuth access token.

    google-auth is an explicit dependency rather than a hand-rolled JWT because
    the assertion must be RS256-signed and the standard library has no RSA. A
    hand-rolled signer here would be a security-sensitive reimplementation of a
    solved problem, shelling out to `openssl` would not survive PowerShell, and
    both would still need refresh handling.
    """
    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request
    except ImportError:
        fail(
            "google-auth is not installed. Create the isolated tooling venv:\n"
            "  py -3 -m venv Tools/publish-venv\n"
            "  Tools/publish-venv/Scripts/python.exe -m pip install "
            "-r Tools/requirements-publish.txt\n"
            "then re-run this script with that interpreter. Do NOT install it "
            "into .venv — that venv's pins are load-bearing for ML-Agents training."
        )

    if not os.path.exists(credentials_path):
        fail(
            "no service account key at " + credentials_path + "\n"
            "See the SETUP block at the top of this file — the key has to be "
            "created and granted release permission by a human in a browser; "
            "there is no way around that step."
        )

    credentials = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=[SCOPE])
    credentials.refresh(Request())
    return credentials.token


def call(token, method, url, body=None, content_type="application/json",
         raw=False):
    data = None
    headers = {"Authorization": "Bearer " + token}
    if body is not None:
        data = body if raw else json.dumps(body).encode("utf-8")
        headers["Content-Type"] = content_type
        headers["Content-Length"] = str(len(data))

    request = urllib.request.Request(url, data=data, headers=headers,
                                     method=method)
    try:
        with urllib.request.urlopen(request) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        # The API's own message is far more useful than the status line —
        # "versionCode 2 has already been used" versus a bare 400.
        fail("Play API {} on {} {}\n{}".format(error.code, method, url, detail))
    except urllib.error.URLError as error:
        fail("could not reach the Play API: {}".format(error.reason))


def describe_track(token, edit_id, track):
    """Print what is already on the track. A versionCode must be strictly higher
    than everything Play has ever seen for this package, and the overwhelmingly
    common failure is bumping from the LOCAL value while the live one is ahead."""
    url = "{}/applications/{}/edits/{}/tracks/{}".format(
        API_ROOT, PACKAGE_NAME, edit_id, track)
    try:
        current = call(token, "GET", url)
    except SystemExit:
        # A track with nothing on it 404s. That is information, not an error.
        print("  track '{}' has no existing releases".format(track))
        return

    releases = current.get("releases", [])
    if not releases:
        print("  track '{}' has no existing releases".format(track))
        return
    for release in releases:
        print("  track '{}' currently: status={} versionCodes={} name={}".format(
            track,
            release.get("status", "?"),
            release.get("versionCodes", []),
            release.get("name", "")))


def main():
    parser = argparse.ArgumentParser(
        description="Upload a signed AAB to Google Play.")
    parser.add_argument("--aab", default=DEFAULT_AAB,
                        help="path to the .aab (default: Builds/Android/PoRacer.aab)")
    parser.add_argument("--credentials",
                        default=os.environ.get("PORACER_PLAY_CREDENTIALS",
                                               DEFAULT_CREDENTIALS),
                        help="service account JSON key")
    parser.add_argument("--track", default="internal",
                        choices=["internal", "alpha", "beta", "production"],
                        help="release track (default: internal — production is opt-in)")
    parser.add_argument("--status", default="draft",
                        choices=["draft", "completed"],
                        help="draft uploads without rolling out (default: draft)")
    parser.add_argument("--notes", default=None,
                        help="release notes (en-US)")
    parser.add_argument("--dry-run", action="store_true",
                        help="do everything except commit, then delete the edit")
    args = parser.parse_args()

    if not os.path.exists(args.aab):
        fail("no AAB at " + args.aab + " — run PoRacer → Build Android AAB first")

    size_mb = os.path.getsize(args.aab) / (1024.0 * 1024.0)
    print("AAB:   {} ({:.1f} MB)".format(args.aab, size_mb))
    print("App:   {}".format(PACKAGE_NAME))
    print("Track: {}  status={}{}".format(
        args.track, args.status, "  [DRY RUN]" if args.dry_run else ""))

    token = access_token(args.credentials)

    edit = call(token, "POST",
                "{}/applications/{}/edits".format(API_ROOT, PACKAGE_NAME))
    edit_id = edit["id"]
    print("Edit:  {}".format(edit_id))

    describe_track(token, edit_id, args.track)

    with open(args.aab, "rb") as handle:
        blob = handle.read()
    upload_url = "{}/applications/{}/edits/{}/bundles?uploadType=media".format(
        UPLOAD_ROOT, PACKAGE_NAME, edit_id)
    bundle = call(token, "POST", upload_url, body=blob,
                  content_type="application/octet-stream", raw=True)
    version_code = bundle.get("versionCode")
    print("Uploaded bundle versionCode={}".format(version_code))

    release = {
        "status": args.status,
        "versionCodes": [str(version_code)],
    }
    if args.notes:
        release["releaseNotes"] = [{"language": "en-US", "text": args.notes}]

    call(token, "PUT",
         "{}/applications/{}/edits/{}/tracks/{}".format(
             API_ROOT, PACKAGE_NAME, edit_id, args.track),
         body={"track": args.track, "releases": [release]})
    print("Track '{}' set to versionCode {} ({})".format(
        args.track, version_code, args.status))

    if args.dry_run:
        call(token, "DELETE", "{}/applications/{}/edits/{}".format(
            API_ROOT, PACKAGE_NAME, edit_id))
        print("PLAY PUBLISH RESULT: Dry run OK — edit discarded, nothing published. "
              "versionCode {} would have gone to '{}' as {}.".format(
                  version_code, args.track, args.status))
        return

    call(token, "POST", "{}/applications/{}/edits/{}:commit".format(
        API_ROOT, PACKAGE_NAME, edit_id))
    print("PLAY PUBLISH RESULT: Committed versionCode {} to '{}' as {}.".format(
        version_code, args.track, args.status))
    if args.status == "draft":
        print("It is a DRAFT — open the Play Console and roll it out when ready.")


if __name__ == "__main__":
    main()

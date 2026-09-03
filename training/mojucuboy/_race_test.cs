// Put MojucuBoy on the grid and enter play mode. Verification continues in
// _race_check.cs once a few seconds of race have actually run.
var sb = new System.Text.StringBuilder();
var s = UnityEditor.SceneManagement.EditorSceneManager.OpenScene(
    "Assets/Scenes/SCN_RACE_FLAT.unity", UnityEditor.SceneManagement.OpenSceneMode.Single);
sb.Append("opened ").Append(s.path).Append('\n');
UnityEditor.EditorPrefs.SetString("MojucuBoyRaceTest", "armed");
UnityEditor.EditorApplication.EnterPlaymode();
return sb.ToString();

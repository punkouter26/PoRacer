var sb = new System.Text.StringBuilder();
var guids = UnityEditor.AssetDatabase.FindAssets("t:CreatureCatalog");
var cat = UnityEditor.AssetDatabase.LoadAssetAtPath<PoRacer.Models.CreatureCatalog>(UnityEditor.AssetDatabase.GUIDToAssetPath(guids[0]));
sb.Append("catalog entries: ").Append(cat.Entries.Count).Append('\n');
foreach (var e in cat.Entries) {
    string flags = "";
    if (e.prefab != null) {
        if (e.prefab.GetComponent<PoRacer.Agents.IMujocoCreature>() != null) flags += " MUJOCO";
        if (e.prefab.GetComponent<PoRacer.Agents.IAuthoredAppearance>() != null) flags += " AUTHORED-ART";
    }
    sb.Append(string.Format("  {0,-16} {1,-12} prefab={2,-5} HasBrain={3,-5}{4}\n",
        e.id, e.displayName, e.prefab != null, e.HasBrain, flags));
}
return sb.ToString();

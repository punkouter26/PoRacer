var projectRoot = System.IO.Directory.GetParent(UnityEngine.Application.dataPath).FullName;
var outPath = System.IO.Path.Combine(projectRoot, "training", "mojucuboy", "_race_start.txt");
var sb = new System.Text.StringBuilder();

// Walk the visual tree by hand: the UQuery extension methods are not in scope in
// the eval host, and manual recursion needs no using directive.
System.Func<UnityEngine.UIElements.VisualElement, UnityEngine.UIElements.Button> find = null;
find = (el) => {
    var b = el as UnityEngine.UIElements.Button;
    if (b != null && b.text != null && b.text.ToUpperInvariant().Contains("START")) { return b; }
    for (int i = 0; i < el.childCount; i++) {
        var hit = find(el[i]);
        if (hit != null) { return hit; }
    }
    return null;
};

UnityEngine.UIElements.Button target = null;
int docs = 0;
foreach (var d in UnityEngine.Object.FindObjectsByType<UnityEngine.UIElements.UIDocument>(UnityEngine.FindObjectsInactive.Include)) {
    docs++;
    if (d.rootVisualElement == null) { continue; }
    if (target == null) { target = find(d.rootVisualElement); }
}
sb.Append("UIDocuments: ").Append(docs).Append('\n');
if (target == null) { System.IO.File.WriteAllText(outPath, sb.Append("START button not found").ToString()); return "no button"; }
sb.Append("button: '").Append(target.text).Append("'\n");

var cf = typeof(UnityEngine.UIElements.Clickable).GetField("clicked", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
var act = cf == null ? null : cf.GetValue(target.clickable) as System.Action;
if (act == null) { System.IO.File.WriteAllText(outPath, sb.Append("could not reach clicked delegate").ToString()); return "no delegate"; }
act.Invoke();
sb.Append("invoked START\n");
System.IO.File.WriteAllText(outPath, sb.ToString());
return sb.ToString();

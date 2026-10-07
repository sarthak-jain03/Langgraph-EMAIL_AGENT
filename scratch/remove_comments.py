import re

with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    stripped = line.strip()
    
    # Skip CSS comments
    if stripped.startswith("/*") and stripped.endswith("*/"):
        continue
        
    # Skip Python comments (starts with #, but make sure it's not #MainMenu inside the CSS)
    # The only # inside CSS that starts a line in our code is #MainMenu {visibility: hidden;}
    if stripped.startswith("#") and not stripped.startswith("#MainMenu"):
        # Exception for # input -> review -> done (which is inline)
        continue
        
    # Handle inline comments like `st.session_state.stage = "input"  # input → review → done`
    if "  # " in line:
        line = line.split("  # ")[0] + "\n"
        
    new_lines.append(line)

with open("app.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

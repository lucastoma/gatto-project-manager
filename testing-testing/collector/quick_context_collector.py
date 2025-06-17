import tkinter as tk
from tkinter import filedialog, ttk
import yaml
from pathlib import Path
import os
import json # Added for history
import fnmatch

CONFIG_FILE_NAME = "config/context_filters.yaml"
HISTORY_FILE_NAME = "config/context_history.json"
MAX_HISTORY_ITEMS = 5

class QuickContextCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quick Context Collector")
        # Adjust initial size to accommodate new elements
        self.root.geometry("500x350")

        self.selected_directory = tk.StringVar()
        self.selected_filter_name = tk.StringVar()
        self.selected_history_entry = tk.StringVar() # For the new history combobox
        self.selected_exclude_pattern = tk.StringVar(value="None") # For the new exclude combobox
        self.save_to_central_dir = tk.BooleanVar(value=True) # For the new Checkbutton
        self.last_output_path = None # To store the path of the last generated file
        self.filters = {}
        self.history = []
        self.script_dir = Path(__file__).parent
        self.workspace_root = self.script_dir.parent # Define workspace_root for central export path
        self.history_file_path = self.script_dir / HISTORY_FILE_NAME

        # --- UI Elements ---
        # Directory Selection
        tk.Label(root, text="Directory:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.dir_entry = tk.Entry(root, textvariable=self.selected_directory, width=50)
        self.dir_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.browse_button = tk.Button(root, text="Browse...", command=self.browse_directory)
        self.browse_button.grid(row=0, column=2, padx=5, pady=5)

        # History Selection
        tk.Label(root, text="History:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.history_combobox = ttk.Combobox(root, textvariable=self.selected_history_entry, state="readonly", width=47)
        self.history_combobox.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.history_combobox.bind("<<ComboboxSelected>>", self.on_history_selected)

        # Filter Selection
        tk.Label(root, text="Filter:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.filter_combobox = ttk.Combobox(root, textvariable=self.selected_filter_name, state="readonly", width=47)
        self.filter_combobox.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        # Exclude Pattern Selection
        tk.Label(root, text="Exclude:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.exclude_combobox = ttk.Combobox(root, textvariable=self.selected_exclude_pattern, state="readonly", width=47)
        self.exclude_combobox['values'] = ["None", "test", "legacy", "test & legacy"]
        self.exclude_combobox.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        # Save Location Checkbutton
        self.save_to_central_dir_checkbutton = tk.Checkbutton(root, text="Save to central export directory (.doc-gen/export)", variable=self.save_to_central_dir)
        self.save_to_central_dir_checkbutton.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # Action Buttons Frame
        action_buttons_frame = tk.Frame(root)
        action_buttons_frame.grid(row=5, column=1, padx=5, pady=10, sticky="ew")
        action_buttons_frame.grid_columnconfigure(0, weight=1)
        action_buttons_frame.grid_columnconfigure(1, weight=1)

        self.collect_button = tk.Button(action_buttons_frame, text="Collect Context", command=self.collect_context, height=2, width=15)
        self.collect_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.copy_button = tk.Button(action_buttons_frame, text="Copy Output", command=self.copy_output_to_clipboard, height=2, width=15)
        self.copy_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Status Bar (optional)
        self.status_label = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=6, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        # Configure grid column weights for resizing
        root.grid_columnconfigure(1, weight=1)

        # Initial Load
        self.load_filters()
        self.load_history()

    def load_filters(self):
        try:
            config_path = self.script_dir / CONFIG_FILE_NAME
            with open(config_path, "r", encoding="utf-8") as f:
                self.filters = yaml.safe_load(f).get("filters", {})
                self.filter_combobox['values'] = list(self.filters.keys())
                self.update_status(f"Loaded {len(self.filters)} filters from {CONFIG_FILE_NAME}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load or parse {CONFIG_FILE_NAME}:\n{e}")
            self.root.quit()

    def browse_directory(self):
        # Start browsing from the workspace root directory
        initial_dir = self.workspace_root
        directory = filedialog.askdirectory(initialdir=initial_dir, title="Select a Directory")
        if directory:
            self.selected_directory.set(directory)

    def collect_context(self):
        directory_str = self.selected_directory.get()
        filter_name = self.selected_filter_name.get()

        if not directory_str or not filter_name:
            self.update_status("Please select a directory and a filter first.")
            return

        target_dir = Path(directory_str)
        if not target_dir.is_dir():
            self.update_status(f"Error: Directory not found at {target_dir}")
            return

        selected_filter_patterns = self.filters[filter_name].get("patterns", ["*.*"])
        base_output_filename = f"{target_dir.name}_context__{filter_name.replace(' (*.*)','').replace('*','all').replace('.','')}.txt"

        if self.save_to_central_dir.get():
            central_export_dir = self.workspace_root / ".doc-gen" / "export"
            central_export_dir.mkdir(parents=True, exist_ok=True)
            output_path = central_export_dir / base_output_filename
        else:
            output_path = target_dir / base_output_filename

        self.save_history(directory_str, filter_name)

        try:
            self.update_status(f"Collecting context... Filter: {filter_name}")

            exclude_option = self.selected_exclude_pattern.get()
            exclude_dirs = {'__pycache__', '.git', '.svn', 'node_modules', '.venv', 'venv'}
            if exclude_option == "test":
                exclude_dirs.update(['test', 'tests'])
            elif exclude_option == "legacy":
                exclude_dirs.add('legacy')
            elif exclude_option == "test & legacy":
                exclude_dirs.update(['test', 'tests', 'legacy'])

            found_files = []
            for root, dirs, files in os.walk(target_dir):
                dirs[:] = [d for d in dirs if d.lower() not in exclude_dirs]
                
                for filename in files:
                    for pattern in selected_filter_patterns:
                        if fnmatch.fnmatch(filename, pattern):
                            found_files.append(Path(root) / filename)
                            break
            
            found_files = sorted(list(set(found_files)))

            if not found_files:
                self.update_status(f"No files found matching the filter in {target_dir.name}")
                return

            all_content = ""
            for file_path in found_files:
                try:
                    relative_path = file_path.relative_to(self.workspace_root)
                    all_content += f"--- START {relative_path} ---\n"
                    all_content += file_path.read_text(encoding='utf-8', errors='ignore')
                    all_content += f"\n--- END {relative_path} ---\n\n"
                except Exception as e:
                    all_content += f"--- ERROR reading {file_path}: {e} ---\n\n"
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(all_content)
            self.last_output_path = output_path
            self.update_status(f"Successfully collected {len(found_files)} files to {output_path.name}")
            print(f"Successfully collected {len(found_files)} files to {output_path}")
        except Exception as e:
            self.last_output_path = None
            self.update_status(f"Error during collection: {e}")
            print(f"Error during collection: {e}")

    def format_history_entry_display(self, entry):
        dir_path = Path(entry.get("directory", "N/A"))
        filter_name = entry.get("filter_name", "N/A")
        return f"{dir_path.name}  |  {filter_name}"

    def update_history_combobox(self):
        display_entries = [self.format_history_entry_display(entry) for entry in self.history]
        self.history_combobox['values'] = display_entries

    def on_history_selected(self, event):
        selected_display_text = self.selected_history_entry.get()
        for entry in self.history:
            if self.format_history_entry_display(entry) == selected_display_text:
                if Path(entry.get("directory", "")).is_dir() and \
                   entry.get("filter_name", "") in self.filters:
                    self.selected_directory.set(entry["directory"])
                    self.selected_filter_name.set(entry["filter_name"])
                    self.update_status(f"Selected from history: {Path(entry['directory']).name} | {entry['filter_name']}")
                else:
                    self.update_status(f"Invalid history entry selected: {selected_display_text}")
                return

    def load_history(self):
        try:
            if self.history_file_path.exists():
                with open(self.history_file_path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
                    if not isinstance(self.history, list):
                        self.history = []
            else:
                 self.history = []
        except Exception as e:
            self.history = []
            self.update_status(f"Error loading history file: {e}. Starting with empty history.")
            print(f"Error loading history file: {e}")

        self.update_history_combobox()

        if self.history:
            most_recent = self.history[0]
            if Path(most_recent.get("directory", "")).is_dir() and \
               most_recent.get("filter_name", "") in self.filters:
                self.selected_directory.set(most_recent["directory"])
                self.selected_filter_name.set(most_recent["filter_name"])
                self.update_status(f"Loaded last used: {Path(most_recent['directory']).name} | {most_recent['filter_name']}")
                return
        
        self.update_status("No valid recent settings or history file not found. Using defaults.")

    def save_history(self, directory_str, filter_name_str):
        if not directory_str or not filter_name_str:
            return

        new_entry = {"directory": directory_str, "filter_name": filter_name_str}
        
        self.history = [entry for entry in self.history if not (entry.get("directory") == directory_str and entry.get("filter_name") == filter_name_str)]
        
        self.history.insert(0, new_entry)
        self.history = self.history[:MAX_HISTORY_ITEMS]

        try:
            with open(self.history_file_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
            self.update_history_combobox()
        except Exception as e:
            self.update_status(f"Error saving history: {e}")
            print(f"Error saving history: {e}")

    def copy_output_to_clipboard(self):
        if self.last_output_path and self.last_output_path.exists():
            try:
                with open(self.last_output_path, "r", encoding="utf-8") as f:
                    content_to_copy = f.read()
                
                self.root.clipboard_clear()
                self.root.clipboard_append(content_to_copy)
                self.update_status(f"Copied content of {self.last_output_path.name} to clipboard.")
                print(f"Copied content of {self.last_output_path.name} to clipboard.")
            except Exception as e:
                self.update_status(f"Error copying to clipboard: {e}")
                print(f"Error copying to clipboard: {e}")
        else:
            self.update_status("No output file generated yet or file not found.")
            print("No output file generated yet or file not found for copying.")

    def update_status(self, message):
        self.status_label.config(text=message)
        print(message)

if __name__ == "__main__":
    root = tk.Tk()
    app = QuickContextCollectorApp(root)
    root.mainloop()

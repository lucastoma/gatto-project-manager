import tkinter as tk
from tkinter import filedialog, ttk
import yaml
from pathlib import Path
import os
import json # Added for history

CONFIG_FILE_NAME = "context_filters.yaml"
HISTORY_FILE_NAME = "context_history.json"
MAX_HISTORY_ITEMS = 5

class QuickContextCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quick Context Collector")
        self.root.geometry("500x300")

        self.selected_directory = tk.StringVar()
        self.selected_filter_name = tk.StringVar()
        self.selected_history_entry = tk.StringVar() # For the new history combobox
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
        # self.filter_combobox.bind("<<ComboboxSelected>>", self.on_filter_selected)

        # Save Location Checkbutton
        self.save_to_central_dir_checkbutton = tk.Checkbutton(root, text="Save to central export directory (.doc-gen/export)", variable=self.save_to_central_dir)
        self.save_to_central_dir_checkbutton.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # Action Buttons Frame
        action_buttons_frame = tk.Frame(root)
        action_buttons_frame.grid(row=4, column=1, padx=5, pady=10, sticky="ew") # Adjusted pady
        action_buttons_frame.grid_columnconfigure(0, weight=1)
        action_buttons_frame.grid_columnconfigure(1, weight=1)

        self.collect_button = tk.Button(action_buttons_frame, text="Collect Context", command=self.collect_context, height=2, width=15)
        self.collect_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew") # pady within frame

        self.copy_button = tk.Button(action_buttons_frame, text="Copy Output", command=self.copy_output_to_clipboard, height=2, width=15)
        self.copy_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew") # pady within frame

        # Status Bar (optional)
        self.status_label = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=5, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        # Configure grid column weights for resizing
        root.grid_columnconfigure(1, weight=1)

        self.load_filters()
        self.update_filter_combobox() # Ensure filters are loaded before trying to match history
        self.load_history() # Load history and apply the most recent selection

    def browse_directory(self):
        # Determine workspace root (parent of the .doc-gen directory where the script resides)
        script_dir = Path(__file__).parent
        workspace_root = script_dir.parent
        
        directory = filedialog.askdirectory(initialdir=str(workspace_root))
        if directory:
            self.selected_directory.set(directory)
            self.update_status(f"Selected directory: {directory}")

    def load_filters(self):
        try:
            script_dir = Path(__file__).parent
            config_path = script_dir / CONFIG_FILE_NAME
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    loaded_config = yaml.safe_load(f)
                    if loaded_config and "filters" in loaded_config:
                        self.filters = loaded_config["filters"]
                        self.update_status(f"Loaded {len(self.filters)} filters from {CONFIG_FILE_NAME}")
                    else:
                        self.filters = {"All Files (*.*)": {"patterns": ["*.*"], "description": "All files"}}
                        self.update_status("Config file format error or no filters. Using default 'All Files'.")
            else:
                self.filters = {"All Files (*.*)": {"patterns": ["*.*"], "description": "All files"}}
                self.update_status(f"{CONFIG_FILE_NAME} not found. Using default 'All Files'.")
        except Exception as e:
            self.filters = {"All Files (*.*)": {"patterns": ["*.*"], "description": "All files"}}
            self.update_status(f"Error loading filters: {e}. Using default 'All Files'.")
            print(f"Error loading filters: {e}")

    def update_filter_combobox(self):
        filter_names = list(self.filters.keys())
        self.filter_combobox['values'] = filter_names
        if filter_names:
            self.selected_filter_name.set(filter_names[0])

    # def on_filter_selected(self, event):
    #     selected_name = self.selected_filter_name.get()
    #     if selected_name in self.filters:
    #         # Potentially update UI or log description
    #         description = self.filters[selected_name].get("description", "No description")
    #         self.update_status(f"Selected filter: {selected_name} ({description})")
    #     pass

    def collect_context(self):
        directory_str = self.selected_directory.get()
        filter_name = self.selected_filter_name.get()

        if not directory_str:
            self.update_status("Error: Please select a directory.")
            return
        if not filter_name or filter_name not in self.filters:
            self.update_status("Error: Please select a valid filter.")
            return

        target_dir = Path(directory_str)
        if not target_dir.is_dir():
            self.update_status(f"Error: Invalid directory: {directory_str}")
            return

        selected_filter_patterns = self.filters[filter_name].get("patterns", ["*.*"])
        base_output_filename = f"{target_dir.name}_context__{filter_name.replace(' (*.*)','').replace('*','all').replace('.','')}.txt"

        if self.save_to_central_dir.get():
            central_export_dir = self.workspace_root / ".doc-gen" / "export"
            central_export_dir.mkdir(parents=True, exist_ok=True) # Ensure central export directory exists
            output_path = central_export_dir / base_output_filename
        else:
            output_path = target_dir / base_output_filename

        # Save current selection to history before processing
        self.save_history(directory_str, filter_name)

        self.update_status(f"Collecting context... Filter: {filter_name}")
        collected_content = []
        file_count = 0

        try:
            for pattern in selected_filter_patterns:
                for filepath in target_dir.rglob(pattern):
                    if filepath.is_file():
                        try:
                            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read()
                            collected_content.append(f"\n--- File: {filepath.relative_to(target_dir)} ---\n")
                            collected_content.append(content)
                            file_count += 1
                        except Exception as e:
                            collected_content.append(f"\n--- Error reading file: {filepath.relative_to(target_dir)}: {e} ---\n")
            
            if collected_content:
                with open(output_path, "w", encoding="utf-8") as outfile:
                    outfile.write("Collected context from directory: " + str(target_dir) + "\n")
                    outfile.write("Filter used: " + filter_name + "\n")
                    outfile.write("Patterns: " + str(selected_filter_patterns) + "\n")
                    outfile.write("Total files processed: " + str(file_count) + "\n")
                    outfile.write("="*80 + "\n")
                    outfile.write("".join(collected_content))
                self.last_output_path = output_path # Store the path of the generated file
                self.update_status(f"Successfully collected {file_count} files to {output_path.name}")
            else:
                self.update_status(f"No files found matching the filter in {target_dir.name}")

        except Exception as e:
            self.last_output_path = None # Reset on error
            self.update_status(f"Error during collection: {e}")
            print(f"Error during collection: {e}")

    def format_history_entry_display(self, entry):
        dir_path = Path(entry.get("directory", "N/A"))
        filter_name = entry.get("filter_name", "N/A")
        return f"{dir_path.name}  |  {filter_name}"

    def update_history_combobox(self):
        display_entries = [self.format_history_entry_display(entry) for entry in self.history]
        self.history_combobox['values'] = display_entries
        if display_entries:
            # Optionally, set the combobox to the most recent entry's display string
            # self.selected_history_entry.set(display_entries[0]) 
            # However, it might be better to leave it blank or with a placeholder initially
            # and let the auto-load fill the main fields.
            pass 

    def on_history_selected(self, event):
        selected_display_text = self.selected_history_entry.get()
        # Find the original history entry that matches the display text
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
                        self.history = [] # Ensure history is a list
            else:
                 self.history = []
        except Exception as e:
            self.history = []
            self.update_status(f"Error loading history file: {e}. Starting with empty history.")
            print(f"Error loading history file: {e}")

        self.update_history_combobox() # Populate the history combobox

        # Apply the most recent valid history item to main fields
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
            return # Do not save empty or invalid entries

        new_entry = {"directory": directory_str, "filter_name": filter_name_str}
        
        # Remove existing identical entry to move it to the top
        self.history = [entry for entry in self.history if not (entry.get("directory") == directory_str and entry.get("filter_name") == filter_name_str)]
        
        self.history.insert(0, new_entry)
        self.history = self.history[:MAX_HISTORY_ITEMS] # Keep only the top N items

        try:
            with open(self.history_file_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
            self.update_history_combobox() # Refresh history combobox after saving
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
        print(message) # Also print to console for debugging

if __name__ == "__main__":
    root = tk.Tk()
    app = QuickContextCollectorApp(root)
    root.mainloop()

import os

def extract_md_to_txt(root_dir, output_file):
    """
    Finds all .md files in a directory and its subdirectories,
    and combines their content into a single .txt file.

    Args:
        root_dir (str): The path to the root directory to search.
        output_file (str): The path to the output text file.
    """
    # A list to hold the content of all markdown files found.
    all_md_content = []
    
    print(f"Starting search in directory: {root_dir}")

    # os.walk() generates the file names in a directory tree,
    # by walking the tree either top-down or bottom-up.
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Check if the file has a .md extension.
            if filename.endswith(".md"):
                # Construct the full file path.
                file_path = os.path.join(dirpath, filename)
                print(f"Found Markdown file: {file_path}")
                
                try:
                    # Open and read the content of the markdown file.
                    with open(file_path, 'r', encoding='utf-8') as md_file:
                        content = md_file.read()
                        
                        # Add a header to distinguish content from different files.
                        header = f"\n\n--- Content from: {file_path} ---\n\n"
                        all_md_content.append(header)
                        all_md_content.append(content)
                        
                except Exception as e:
                    # Handle potential reading errors.
                    error_message = f"\n\n--- Error reading file: {file_path} --- \n{e}\n"
                    all_md_content.append(error_message)
                    print(error_message)

    # Write the combined content to the output file.
    try:
        with open(output_file, 'w', encoding='utf-8') as txt_file:
            txt_file.write("".join(all_md_content))
        print(f"\nSuccessfully combined all .md files into: {output_file}")
    except Exception as e:
        print(f"\nError writing to output file: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    # Set the root directory to the 'docs' folder inside the cloned repository.
    # You might need to change this path depending on where you run the script from.
    source_directory = "gemini-cli/docs"
    
    # Set the name for the final output text file.
    destination_file = "gemini_cli_docs.txt"
    
    # Check if the source directory exists before running.
    if not os.path.isdir(source_directory):
        print(f"Error: The source directory '{source_directory}' does not exist.")
        print("Please make sure you have cloned the repository and are running the script from the correct location.")
    else:
        # Run the function.
        extract_md_to_txt(source_directory, destination_file)

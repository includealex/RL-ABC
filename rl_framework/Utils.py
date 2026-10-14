"""Canonical Elegant preprocessing and RL observation utilities.

This module keeps the established public helper API while using the hardened
lattice preprocessing, graph-building, observation, and validation functions.

Design rules
------------
* Comments are never parsed as active Elegant statements and are copied intact.
* ``&`` continuation lines are joined for parsing.
* ``LINE`` expressions are parsed structurally, including ``N*item`` and
  reversed ``-item`` references.
* The selected beamline is flattened to base-element occurrences.
* Every repeated controllable element gets an occurrence-specific definition
  (``NAME_1``, ``NAME_2``, ...); passive elements keep their original names.
* A generated WATCH is placed immediately before every active element that has
  at least one existing controllable parameter: K1, HKICK, VKICK, or FSE.
* Only those existing controllable values are changed.  The reset lattice gets
  zeroes, while the returned template gets agent-value placeholders.  All other
  parameters, including TILT, are preserved verbatim.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Compatibility helpers appear first; hardened lattice functions later in this
# file are the canonical definitions exported by Python.
# ---------------------------------------------------------------------------

import re
import os
import shutil
import subprocess
from collections import deque
import torch
import numpy as np
import random


def set_seed(seed):
    """
    Set seed for all libraries to ensure reproducibility.
    
    Parameters:
    - seed (int): The seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed set to: {seed}")


def _normalize_ele_stem(input_beam_file):
    """Return .ele file stem without extension."""
    return input_beam_file[:-4] if str(input_beam_file).endswith(".ele") else input_beam_file


def update_lattice_filename_in_ele(input_beam_file, elegant_input_filename):
    """
    Updates the lattice filename in the .ele tracking file.
    """
    input_beam_file = _normalize_ele_stem(input_beam_file)
    ele_file_path = input_beam_file + ".ele"
    
    if not os.path.exists(ele_file_path):
        print(f"Warning: {ele_file_path} does not exist. Skipping lattice filename update.")
        return
    
    try:
        with open(ele_file_path, 'r') as f:
            content = f.read()
        
        updated_content = re.sub(
            r'(\s*lattice\s*=\s*).*?(\s*(?:&|\n|$))',
            rf'\1{elegant_input_filename}\2',
            content,
            flags=re.IGNORECASE
        )
        
        with open(ele_file_path, 'w') as f:
            f.write(updated_content)
        
        print(f"Updated lattice filename in {ele_file_path} to: {elegant_input_filename}")
    
    except Exception as e:
        print(f"Error updating lattice filename in {ele_file_path}: {e}")


def update_results_path_in_ele(input_beam_file, results_path):
    """
    Updates all results folder references in a .ele tracking file.
    """
    input_beam_file = _normalize_ele_stem(input_beam_file)
    ele_file_path = input_beam_file + ".ele"
    if not os.path.exists(ele_file_path):
        print(f"Warning: {ele_file_path} does not exist. Skipping results path update.")
        return

    results_folder = str(results_path).rstrip("/\\")
    if not results_folder:
        return

    try:
        with open(ele_file_path, "r") as f:
            content = f.read()

        updated_content = re.sub(
            r'(?<![\w/])results(?:_[^/\s"]+)?/',
            f"{results_folder}/",
            content
        )

        with open(ele_file_path, "w") as f:
            f.write(updated_content)

    except Exception as e:
        print(f"Error updating results path in {ele_file_path}: {e}")


def update_random_seed_in_ele(input_beam_file, seed):
    """
    Updates random_number_seed in a .ele tracking file.
    """
    input_beam_file = _normalize_ele_stem(input_beam_file)
    ele_file_path = input_beam_file + ".ele"
    if not os.path.exists(ele_file_path):
        print(f"Warning: {ele_file_path} does not exist. Skipping random seed update.")
        return

    try:
        with open(ele_file_path, "r") as f:
            content = f.read()

        updated_content = re.sub(
            r"(\s*random_number_seed\s*=\s*)\d+",
            rf"\g<1>{int(seed)}",
            content,
            flags=re.IGNORECASE
        )

        with open(ele_file_path, "w") as f:
            f.write(updated_content)

    except Exception as e:
        print(f"Error updating random seed in {ele_file_path}: {e}")


def create_seeded_tracking_file(input_beam_file, elegant_input_filename, results_path, seed):
    """
    Creates a seed-specific copy of the tracking .ele file and updates:
    - lattice filename
    - results folder paths
    - random_number_seed

    Returns:
    - seeded_input_beam_file (str): file stem without extension, e.g. 'track_800'
    """
    input_beam_file = _normalize_ele_stem(input_beam_file)
    source_ele_path = f"{input_beam_file}.ele"
    seeded_input_beam_file = f"{input_beam_file}_{seed}"
    seeded_ele_path = f"{seeded_input_beam_file}.ele"

    if not os.path.exists(source_ele_path):
        raise FileNotFoundError(f"Tracking file not found: {source_ele_path}")

    shutil.copy2(source_ele_path, seeded_ele_path)
    update_lattice_filename_in_ele(seeded_input_beam_file, elegant_input_filename)
    update_results_path_in_ele(seeded_input_beam_file, results_path)
    update_random_seed_in_ele(seeded_input_beam_file, seed)

    return seeded_input_beam_file


#remove alll watch points that we didn't add
# and update the line definitions accordingly
# Save the original file content into a backup file (seed-specific when backup_path is set).
# This function removes all 'watch' elements from the .lte file and updates line definitions accordingly.
def remove_watch_points(file_path, backup_path="unchanged_input.lte"):
    """
    Removes all 'watch' elements from the .lte file and updates line definitions accordingly.
    Saves the original file content into *backup_path* before modifying *file_path*.

    :param file_path: Path to the .lte file
    :param backup_path: Path for the pre-modification backup copy
    """
    # Save the original file content
    with open(file_path, 'r') as file:
        original_content = file.read()
    with open(backup_path, 'w') as backup_file:
        backup_file.write(original_content)

    # Read the file and process its content
    with open(file_path, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    watch_elements = set()  # To store all watch element names

    # First pass: Remove watch elements and collect their names
    for line in lines:
        if re.match(r"^\s*\w+\s*:\s*watch", line, re.IGNORECASE):
            match = re.match(r"^\s*(\w+)\s*:", line)
            if match:
                watch_elements.add(match.group(1))
        else:
            updated_lines.append(line)

    # Second pass: Update line definitions to remove watch elements
    final_lines = []
    for line in updated_lines:
        line_match = re.match(r"^\s*(\w+)\s*:\s*LINE\s*=\s*\((.*)\)", line, re.IGNORECASE)
        if line_match:
            line_name, elements = line_match.groups()
            elements_list = [e.strip() for e in elements.split(",")]
            filtered_elements = [e for e in elements_list if e not in watch_elements]
            final_lines.append(f"{line_name}: LINE=({', '.join(filtered_elements)})")
        else:
            final_lines.append(line.strip())

    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write("\n".join(final_lines) + "\n")


def parse_lattice_file(file_path):
    """Parses a lattice file and extracts relevant elements into a dictionary."""
    elements = {}

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("!"):  # Skip empty lines and comments
                continue
            
            match = re.match(r"(\w+):\s*(\w+),\s*(.*)", line)
            if match:
                name, element_type, params = match.groups()

                # Identify element type
                if "WATCH" in element_type.upper():
                    category = "watch point"
                elif "QUAD" in element_type.upper():
                    category = "quadrupole magnet"
                elif "SBEND" in element_type.upper():
                    category = "dipole magnet"
                elif "DRIFT" in element_type.upper():
                    category = "drift"
                elif "MAXAMP" in element_type.upper():  
                    continue  # Skip apertures

                else:
                    continue  # Skip other elements

                # Extract parameters
                param_dict = {}
                for param in params.split(","):
                    key_value = param.split("=")
                    if len(key_value) == 2:
                        key, value = key_value
                        param_dict[key.strip()] = value.strip()
                
                elements[name] = {"type": category, "parameters": param_dict}

    return elements



'''
1. Watch points are only added after QUAD and SBEND elements.
2. The watch points are stored correctly and used sequentially for each repeated element instance.
3. LINE definitions are updated with the correct watch points in their correct positions.
'''

def add_watch_points(input_file, output_file, results_path):
    """
    Adds a watch point after each repeated quadrupole (QUAD) or dipole (SBEND) magnet in the lattice file,
    while preserving and updating LINE definitions to include only the generated watch points.
    """
    element_counts = {}  # Store element counts for QUAD and SBEND elements
    line_definitions = {}  # Store LINE definitions
    updated_lines = []  # Store intermediate lines without modification
    watch_points = []  # Store generated watch points
    controllable_elements = []  # Elements to add watch points after (QUAD, SBEND)
    watch_point_queues = {}  # Dictionary mapping elements to queues of watch points

    # First pass: Identify and count occurrences of QUAD and SBEND elements in LINE definitions
    with open(input_file, "r") as file:
        for line in file:
            line = line.rstrip()
            
            # Capture LINE definitions
            line_match = re.match(r"(\w+):\s*LINE\s*=\s*\((.*)\)", line)
            if line_match:
                line_name, elements = line_match.groups()
                elements_list = [e.strip() for e in elements.split(",")]
                line_definitions[line_name] = elements_list
                
                # Count occurrences only for QUAD and SBEND elements
                for element in elements_list:
                    if re.match(r"(\w+)", element):
                        element_counts[element] = element_counts.get(element, 0) + 1
                continue  # Store for later modification
            
            updated_lines.append(line)
    
    # Second pass: Modify the input file and add watch points
    final_lines = []
    
    for line in updated_lines:
        line = line.rstrip()
        
        # Match QUAD or SBEND elements
        match = re.match(r"(\w+):\s*(QUAD|SBEND)", line)
        if match:
            element_name = match.group(1)
            controllable_elements.append(element_name)
            
            # Add watch points based on element count
            if element_name in element_counts:
                count = element_counts[element_name]
                watch_point_queues[element_name] = deque()
                
                for i in range(1, count + 1):
                    watch_name = f"W{element_name}_{i}"
                    watch_line = f"{watch_name}: WATCH, filename=\"{results_path}{watch_name}.sdds\", mode=coord"
                    watch_points.append(watch_name)
                    final_lines.append(watch_line)
                    watch_point_queues[element_name].append(watch_name)
        
        final_lines.append(line)
    
    # Update LINE definitions with new watch points in correct positions
    for line_name, elements in line_definitions.items():
        new_elements_list = []
        
        for element in elements:
            if element in watch_point_queues and watch_point_queues[element]:
                new_elements_list.append(watch_point_queues[element].popleft())
            new_elements_list.append(element)
        
        final_lines.append(f"{line_name}: LINE=({', '.join(new_elements_list)})")
    
    # Write the updated lines to the output file
    with open(output_file, "w") as file:
        file.write("\n".join(final_lines) + "\n")

def add_final_watch_point(input_file, output_file, beamline_name,results_path):
    """
    Adds a new watch point named 'final_WP' to the specified beamline in the lattice file.
    The new watch point is defined at the beginning of the file.

    Parameters:
        input_file (str): Path to the input file containing the lattice definitions.
        output_file (str): Path to the output file where the modified content will be saved.
        beamline_name (str): The name of the beamline to which the watch point should be added.
    """
    # Define the new watch point
    new_watch_point = f'final_WP: WATCH, filename="{results_path}final_WP.sdds", mode=coord'

    # Read the input file
    with open(input_file, "r") as file:
        lines = file.readlines()

    # Initialize variables
    updated_lines = []
    beamline_updated = False
    watch_point_added = False

    # Process each line
    for i, line in enumerate(lines):
        line = line.rstrip()

        # Add the new watch point definition at the beginning of the file
        if not watch_point_added and not line.startswith("!"):
            updated_lines.append(new_watch_point)
            watch_point_added = True

        # Check if the line defines the specified beamline
        if line.startswith(f"{beamline_name}: LINE=(") and not beamline_updated:
            # Find the closing parenthesis and append 'final_WP' before it
            closing_paren_index = line.rfind(")")
            if closing_paren_index != -1:
                line = line[:closing_paren_index] + ", final_WP" + line[closing_paren_index:]
                beamline_updated = True

        updated_lines.append(line)

    # Write the updated content to the output file
    with open(output_file, "w") as file:
        file.write("\n".join(updated_lines) + "\n")

def change_initial_content(file_path, output_file):
    variables = []
    # Read the input file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Patterns to identify controllable elements and attributes
    quad_pattern = re.compile(r"(\w+)\s*:\s*QUAD\s*,\s*L\s*=\s*([\d\.eE+-]+)\s*,\s*K1\s*=\s*([\d\.eE+-]+)\s*,\s*HKICK\s*=\s*([\d\.eE+-]+)\s*,\s*VKICK\s*=\s*([\d\.eE+-]+)")#r"^(w+):\s*QUAD,.*?K1\s*=\s*([-\d.eE]+),\s+HKICK\s*=\s*([-\d.eE]+),\s+VKICK=([-\d.eE]+)")
    sbend_pattern = re.compile(r"(\w+)\s*:\s*SBEND\s*,\s*L\s*=\s*([\d\.eE+-]+)\s*,\s*ANGLE\s*=\s*([\d\.eE+-]+)(?:\s*,\s*FSE\s*=\s*([\d\.eE+-]+))?")

    # Store modified lines
    modified_lines = []

    for line in lines:
        # Check for QUAD elements and modify K1, HKICK, and VKICK
        quad_match = quad_pattern.search(line)
        if quad_match:
            element, L, K1, HKICK, VKICK  = quad_match.groups()
            new_line = (
                f"{element}:  QUAD,L={L}, "
                f"K1={{variables['{element}K1']}}, "
                f"HKICK={{variables['{element}HKICK']}}, "
                f"VKICK={{variables['{element}VKICK']}}\n"
            )
            modified_lines.append(new_line)
            variables.append(f"{element}K1")
            variables.append(f"{element}HKICK")
            variables.append(f"{element}VKICK")
            continue

        # Check for SBEND elements and modify FSE
        sbend_match = sbend_pattern.search(line)
        if sbend_match:
            element, L, ANGLE, FSE = sbend_match.groups()
            new_line = (
                f"{element}:    SBEND,  L={L},      ANGLE={ANGLE}, "
                f"FSE={{variables['{element}FSE']}}\n"
            )
            modified_lines.append(new_line)
            variables.append(f"{element}FSE")
            continue

        # If no match, keep the line unchanged
        modified_lines.append(line)

    # Join all lines to form the final content
    modified_content = "".join(modified_lines)

    with open(output_file, "w") as file:
        file.write("\n".join(modified_lines) + "\n")
    return modified_content, variables

def create_dict_from_lists(keys, values):
    """
    Creates a dictionary from two lists: one for keys and one for values.
    If the lists have different lengths, the extra elements are ignored.
    
    :param keys: List of keys for the dictionary
    :param values: List of values for the dictionary
    :return: Dictionary with keys from the first list and values from the second list
    """
    # Use zip to pair keys and values, then convert to a dictionary
    return dict(zip(keys, values))


####
#functions for creating a graph from the lattice file
####

import re
from collections import deque, defaultdict

def parse_lte_file(file_path):
    """
    Parses the .lte file and returns a dictionary of element definitions.
    """
    element_definitions = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('!'):
                match = re.match(r"(\w+):\s*(.*)", line)
                if match:
                    element_name, element_definition = match.groups()
                    element_definitions[element_name] = element_definition
    return element_definitions

def expand_beamline(element_definitions, beamline_name):
    """
    Recursively expands the beamline definition to include only base elements.
    """
    def expand(element):
        if element in element_definitions:
            definition = element_definitions[element]
            if definition.startswith('LINE'):
                elements = re.findall(r"\w+", definition)
                expanded_elements = []
                for elem in elements:
                    expanded_elements.extend(expand(elem))
                return expanded_elements
            else:
                return [(element, definition)]
        else:
            return [(element, '')]

    expanded_beamline = expand(beamline_name)
    return expanded_beamline

def create_graph(expanded_beamline):
    """
    Creates a graph-like structure from the expanded beamline.
    """
    r"""graph = []
    for element, definition in expanded_beamline:
        element_type = definition.split(',')[0] if definition else 'UNKNOWN'
        attributes = {}
        if definition:
            attributes = dict(re.findall(r"(\w+)=([\w\.\+\-]+)", definition))
        graph.append({
            'name': element,
            'type': element_type,
            'attributes': attributes
        })"""
    graph = []
    for element in expanded_beamline:
        element_name = element[0] if element else 'UNKNOWN'
        #print("element_name", element_name)
        attributes = {}
        if element[1]:
            # Replace tab characters before extracting attributes
            element_list = element[1].split(',')
            element_type= element_list[0] if element_list[0] else 'UNKNOWN'
            element_attributes =element_list[1:]
            clean_str = str(element_attributes).replace('\\t', '')
            
            attributes = dict(re.findall(r"(\w+)\s*=\s*([\w\.\+\-eE]+)", clean_str))
        else:
            element_type= "UNKNOWN"
            attributes= None
        graph.append({
            'name': element_name,
            'type': element_type,
            'attributes': attributes
        })
    return graph

def process_lte_file_to_graph(file_path, beamline_name):
    """
    Processes the .lte file and returns the expanded beamline graph.
    """
    element_definitions = parse_lte_file(file_path)
    expanded_beamline = expand_beamline(element_definitions, beamline_name)
    graph = create_graph(expanded_beamline)
    return graph

#####
#functions for creating a feature matrix from the graph
# the main issue is the size of the feature matrix depends on the number of unique element
#####
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

'''
Explanation:
1.	Define a Fixed Set of Features: We define a fixed set of possible element types and initialize a OneHotEncoder to encode these types as numerical vectors.
2.	Collect Attribute Names: We define the attribute names that we care about, such as 'L', 'K1', 'HKICK', 'VKICK', 'ANGLE', 'FSE', 'X_MAX', 'Y_MAX', and 'ELLIPTICAL'.
3.	Encode Element Types: For each element in the graph, we use the OneHotEncoder to encode the element type as a numerical vector.
4.	Normalize Numerical Attributes: We collect the attribute values for each element and normalize them using a MinMaxScaler.
5.	Create the Feature Matrix: We combine the encoded element types and normalized attributes into a single feature matrix.
6.	Example Usage: We demonstrate how to use the create_feature_matrix function with a sample graph.
This implementation should provide a feature matrix that can be used as input to a neural network. Each row in the matrix represents an element, and each column represents a feature. The feature matrix can be further processed or fed directly into a neural network for training or inference.
'''

def create_feature_matrix(graph):
    # Define the set of possible element types
    element_types = ['charge', 'WATCH', 'MAXAMP', 'rfca','QUAD', 'SBEND', 'DRIFT', 'UNKNOWN']
    
    # Initialize the OneHotEncoder for element types
    type_encoder = OneHotEncoder(categories=[element_types], handle_unknown='ignore')
    type_encoder.fit(np.array(element_types).reshape(-1, 1))
    
    # Initialize the MinMaxScaler for numerical attributes
    scaler = MinMaxScaler()
    
    # Collect all attribute names
    attribute_names = ['L', 'K1', 'HKICK', 'VKICK', 'ANGLE', 'FSE', 'X_MAX', 'Y_MAX', 'ELLIPTICAL']
    
    # Create the feature matrix
    feature_matrix = []
    for node in graph:
        # Encode the element type
        element_type = node['type']
        type_vector = type_encoder.transform([[element_type]]).toarray().flatten()
        
        # Collect and normalize the attributes
        attributes = []
        for attr in attribute_names:
            value = node['attributes'].get(attr, 0)
            attributes.append(float(value))
        
        # Combine the type vector and attributes
        feature_vector = np.concatenate([type_vector, attributes])
        feature_matrix.append(feature_vector)
    
    # Convert the feature matrix to a numpy array
    feature_matrix = np.array(feature_matrix)
    
    # Normalize the numerical attributes
    if feature_matrix.shape[1] > len(element_types):
        feature_matrix[:, len(element_types):] = scaler.fit_transform(feature_matrix[:, len(element_types):])
    
    return feature_matrix


# To preprocess the observation data and compute the covariance matrix of the standardized data.
# The function should return the covariance matrix of the standardized data and the column means of the original observation data.
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def compute_covariance_matrix_mean(observation):
    """
    Computes the covariance matrix of the standardized observation data,
    handling cases where the number of observations is too small.

    Parameters:
    observation (pd.DataFrame): The observation data as a DataFrame.

    Returns:
    tuple: (covariance matrix, column means)
    """
    x, features = observation.shape

    if x < 3:  # Ensure we have enough samples
        print(f"Warning: Too few samples ({x}). Returning identity matrix as fallback.")
        return np.identity(features), observation.mean()

    # Standardize the data
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(observation), columns=observation.columns)

    # Compute covariance matrix
    cov_matrix_standardized = df_standardized.cov()

    # Add a small regularization term to avoid singular matrix issues
    cov_matrix_standardized += np.eye(features) * 1e-6

    return cov_matrix_standardized, observation.mean()


###
#functions for logging the environment
####
import csv

def setLogger(load, log_file= 'environment_log.csv', headers= ('reward', 'actions', 'duration')):
    """
    Set up the logger for the environment.
    """
    if load== True:
        file_handler = open(log_file, "at")
        logger = csv.DictWriter(file_handler, fieldnames=headers)
    else:
        file_handler = open(log_file, "wt")
        logger = csv.DictWriter(file_handler,
                                  fieldnames= headers )
        logger.writeheader()
    file_handler.flush()
    return logger, file_handler

def reset_specific_keys(input_dict, reset_keys=None):
    """
    Set configured control-variable suffixes to zero.

    ``reset_keys=None`` preserves the historical HKICK/VKICK/FSE default.

    Args:
        input_dict (dict): The input dictionary.

    Returns:
        dict: The updated dictionary with specific keys reset to zero.
    """
    from .Utils_magnet_actions import normalize_reset_keys

    normalized = normalize_reset_keys(reset_keys)
    for key in input_dict.keys():
        if any(str(key).upper().endswith(parameter) for parameter in normalized):
            input_dict[key] = 0
    return input_dict



# for generalizing over the number of initial particles, this function changes the initial number of particles everytime we reset the environment
def change_num_initial_particles(path="", num=10000):
    with open(path, "r") as f:
        lines = f.readlines()
    k = None
    pattern = re.compile(r"^\s*n_particles_per_bunch\s*=\s*(\d+)")
    for i, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            k = int(match.group(1))
            lines[i] = re.sub(r"(\d+)", str(num), line, count=1)
            break
    with open(path, "w") as f:
        f.writelines(lines)
    return k, num


def _active_elegant_text(path):
    """Return an Elegant command file with ``!`` comments removed."""
    with open(path, "r") as stream:
        return "\n".join(line.split("!", 1)[0] for line in stream)


def _namelist_blocks(content, name):
    return re.findall(
        rf"&{re.escape(name)}\b(.*?)&end\b",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _namelist_string(block, parameter):
    match = re.search(
        rf"\b{re.escape(parameter)}\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^,\s]+))",
        block,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return next(value for value in match.groups() if value is not None)


def count_sdds_particles(path, sdds2stream_path=None):
    """Count particles in an Elegant SDDS beam using its mandatory x column."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SDDS input beam file not found: {path}")

    executable = sdds2stream_path
    if not executable or not os.path.isfile(executable):
        executable = shutil.which("sdds2stream")
    if not executable:
        raise FileNotFoundError(
            "Cannot count the SDDS beam: sdds2stream was not found. "
            "Set platform.elegant_path to the SDDS executable directory."
        )

    try:
        result = subprocess.run(
            [executable, path, "-columns=x"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown SDDS error").strip()
        raise RuntimeError(f"Could not read SDDS input beam {path}: {detail}") from exc

    count = sum(1 for line in result.stdout.splitlines() if line.strip())
    if count <= 0:
        raise ValueError(f"SDDS input beam contains no particles: {path}")
    return count


def configure_initial_particles(
    ele_path,
    set_initial_particles,
    configured_count=None,
    sdds2stream_path=None,
):
    """Validate the beam mode and return its authoritative particle count.

    ``set_initial_particles=True`` is the generated-beam mode.  It requires one
    ``&bunched_beam`` namelist and updates ``n_particles_per_bunch``.

    ``set_initial_particles=False`` is the external-beam mode.  It requires one
    ``&sdds_beam`` namelist and counts the particles in that namelist's input
    file.  ``configured_count`` is deliberately ignored in this mode.
    """
    if not isinstance(set_initial_particles, bool):
        raise TypeError("environment.set_initial_particles must be true or false")

    content = _active_elegant_text(ele_path)
    generated_blocks = _namelist_blocks(content, "bunched_beam")
    external_blocks = _namelist_blocks(content, "sdds_beam")

    if generated_blocks and external_blocks:
        raise ValueError(
            f"{ele_path} contains both &bunched_beam and &sdds_beam; "
            "select exactly one beam source"
        )

    if set_initial_particles:
        if external_blocks:
            raise ValueError(
                "environment.set_initial_particles=true cannot be used with "
                f"an &sdds_beam input ({ele_path}). Set it to false."
            )
        if len(generated_blocks) != 1:
            raise ValueError(
                "environment.set_initial_particles=true requires exactly one "
                f"&bunched_beam namelist in {ele_path}"
            )
        if isinstance(configured_count, bool) or not isinstance(configured_count, int):
            raise TypeError(
                "environment.init_num_particles must be a positive integer "
                "when set_initial_particles=true"
            )
        if configured_count <= 0:
            raise ValueError(
                "environment.init_num_particles must be greater than zero "
                "when set_initial_particles=true"
            )
        old_count, actual_count = change_num_initial_particles(
            ele_path, configured_count
        )
        if old_count is None:
            raise ValueError(
                f"The &bunched_beam namelist in {ele_path} has no "
                "n_particles_per_bunch parameter"
            )
        return actual_count, "generated bunched_beam", None

    if generated_blocks:
        raise ValueError(
            "environment.set_initial_particles=false cannot be used with "
            f"a generated &bunched_beam ({ele_path}). Set it to true."
        )
    if len(external_blocks) != 1:
        raise ValueError(
            "environment.set_initial_particles=false requires exactly one "
            f"&sdds_beam namelist in {ele_path}"
        )

    beam_name = _namelist_string(external_blocks[0], "input")
    if not beam_name:
        raise ValueError(f"The &sdds_beam namelist in {ele_path} has no input file")
    beam_path = beam_name
    if not os.path.isabs(beam_path):
        beam_path = os.path.join(os.path.dirname(os.path.abspath(ele_path)), beam_path)
    beam_path = os.path.normpath(beam_path)
    actual_count = count_sdds_particles(beam_path, sdds2stream_path)
    return actual_count, "external sdds_beam", beam_path

#expected sources ["Random", "Memory", "Values"]
def run_episode(env, source= "Random", actions= None, memory_episode= None):
    state = env.reset()
    done = False
    if source== "Random":
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _, info = env.step(action)
            state= next_state
            print(info['output_file'] ,reward, done, env.get_number_of_particles())
            if done:
                break
    elif source== "Memory":
        if memory_episode is not None:
            for i in range(len(memory_episode)):
                next_state, reward, done, _, info = env.step(list(memory_episode[i].action), convert=True)
                state= next_state
                print(info['output_file'] ,reward, done, env.get_number_of_particles())
                if done:
                    break
        else:
            print("No memory episode found.")
    elif source== "Values":
        if actions is not None:
            i=0  
            while not done:
                counts = env._check_number_of_variables_to_be_set_at_this_iteration()
                count = counts[0] if isinstance(counts, tuple) else counts
                channel_index = {"K1": 0, "HKICK": 1, "VKICK": 2, "FSE": 3}
                action = np.zeros(4, dtype=np.float32)
                variables = env._variables_for_iteration(0)
                if isinstance(variables, (list, tuple)):
                    if len(variables) != count:
                        raise RuntimeError(
                            f"Expected {count} variables for the current magnet, "
                            f"found {len(variables)}"
                        )
                    for value, variable in zip(actions[i : i + count], variables):
                        _, parameter = env._split_variable_name(variable)
                        if parameter not in channel_index:
                            raise RuntimeError(f"Unknown controllable variable: {variable}")
                        action[channel_index[parameter]] = value
                elif count == 1:
                    action[3] = actions[i]
                else:
                    action[:count] = actions[i : i + count]
                i= i+ count
                next_state, reward, done, _, info = env.step(action)
                print(info['output_file'] ,info['number_of_particles'],reward, done, env.get_number_of_particles())
                state= next_state
                """if done:
                    break"""
        else:
            print("No actions found.")
    else:
        print("Invalid source.")


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def find_maxamp_for_watch_points(element, graph, watch_points):
    """
    Find the MAXAMP element immediately preceding a given watch point in the graph.

    Parameters:
    - element (str): Name of the watch point (e.g., 'WQ1L0_1').
    - graph (list): List of dictionaries, each containing 'name', 'type', and 'attributes'.
    - watch_points (list): List of watch point names in chronological order.

    Returns:
    - dict or None: The MAXAMP element (dictionary) immediately before the watch point,
      or None if no MAXAMP is found or the element is invalid.
    """
    if element not in watch_points:
        return None

    last_maxamp = None
    for item in graph:
        if item['type'] == 'MAXAMP':
            last_maxamp = item
        if item['name'] == element and item['type'] == 'WATCH':
            return last_maxamp
    return None

'''def find_next_maxamp_for_watch_points(element, graph, watch_points):
    """
    Find the MAXAMP element immediately preceding a given watch point in the graph.

    Parameters:
    - element (str): Name of the watch point (e.g., 'WQ1L0_1').
    - graph (list): List of dictionaries, each containing 'name', 'type', and 'attributes'.
    - watch_points (list): List of watch point names in chronological order.

    Returns:
    - dict or None: The MAXAMP element (dictionary) immediately before the watch point,
      or None if no MAXAMP is found or the element is invalid.
    """
    if element not in watch_points:
        return None

    Found= False
    watch= False
    i=0
    next_maxamp = None
    while not Found:
        item= graph[i]
        if item['name'] == element and item['type'] == 'WATCH':
            watch= True

        if item['type'] == 'MAXAMP' and watch:
            next_maxamp = item
            Found= True
        i=i+1

        return next_maxamp
    return None'''

def find_next_maxamp_for_watch_points(element, graph, watch_points):
    """
    Find the MAXAMP element immediately preceding a given watch point in the graph.

    Parameters:
    - element (str): Name of the watch point (e.g., 'WQ1L0_1').
    - graph (list): List of dictionaries, each containing 'name', 'type', and 'attributes'.
    - watch_points (list): List of watch point names in chronological order.

    Returns:
    - dict or None: The MAXAMP element (dictionary) immediately before the watch point,
      or None if no MAXAMP is found or the element is invalid.
    """
    if element not in watch_points:
        return None, None

     # Track the last MAXAMP element encountered
    last_maxamp = None
    watch= False
    next_maxamp = None

    
    for item in graph:

        # Check if the current item is a MAXAMP
        if item['type'] == 'MAXAMP' and not watch:
            last_maxamp = item
        # Check if the current item is the target watch point
        if item['name'] == element and item['type'] == 'WATCH':
            watch= True
            continue

        if watch and item['type'] == 'WATCH':
            next_maxamp = last_maxamp
            return last_maxamp, next_maxamp

        if (item['type'] == 'MAXAMP' ) and watch:
            next_maxamp = item
            return last_maxamp, next_maxamp
        
    if watch and item['name'] == 'final_WP':
        return last_maxamp, last_maxamp
     # Return None if the watch point is not found or no MAXAMP precedes it
    return None, None

def create_nn_representation(df, a, b, n_bins=5):
    """
    Create a fixed-size feature vector for neural network input from particle data.
    Combines robust statistical summaries, 2D histogram of (x, y), point count,
    fraction of points inside an elliptical region, covariance matrix for x, y, xp, yp,
    and ellipse parameters. Handles empty or small datasets and is robust to outliers.

    Parameters:
    - df: pandas DataFrame with columns ['x', 'y', 'xp', 'yp']
    - a: semi-major axis of the ellipse (x-axis)
    - b: semi-minor axis of the ellipse (y-axis)
    - n_bins: number of bins per dimension for 2D histogram (default: 5)

    Returns:
    - 1D numpy array with:
        - Median, IQR, 10th, 90th percentiles for x, y, xp, yp (4 * 4 = 16)
        - 2D histogram bin counts for (x, y) (n_bins * n_bins = 25 for n_bins=5)
        - Number of points (1)
        - Fraction of points inside ellipse (1)
        - Covariance matrix upper triangle for x, y, xp, yp (10 elements)
        - Ellipse parameters a, b (2)
        Total size: 16 + 25 + 1 + 1 + 10 + 2 = 55
    """
    output_size = 16 + n_bins**2 + 1 + 1 + 10 + 2
    if df.empty:
        return np.zeros(output_size)

    features = ['x', 'y', 'xp', 'yp']
    data = df[features].to_numpy()

    stats = []
    for i in range(data.shape[1]):
        col = data[:, i]
        if len(col) > 0:
            median = np.median(col)
            p10, p90 = np.percentile(col, [10, 90])
            iqr = np.percentile(col, 75) - np.percentile(col, 25) if len(col) > 1 else 0.0
        else:
            median, iqr, p10, p90 = 0.0, 0.0, 0.0, 0.0
        stats.extend([median, iqr, p10, p90])

    if len(df) > 0:
        x_range = np.percentile(df['x'], [1, 99]) if df['x'].std() > 0 else [-1, 1]
        y_range = np.percentile(df['y'], [1, 99]) if df['y'].std() > 0 else [-1, 1]
        hist, xedges, yedges = np.histogram2d(
            df['x'], df['y'], bins=n_bins, range=[x_range, y_range]
        )
        hist = hist.flatten() / len(df) if len(df) > 0 else np.zeros(n_bins**2)
    else:
        hist = np.zeros(n_bins**2)

    point_count = len(df)
    fraction_inside = points_in_region(df, a, b) / 100 if not df.empty else 0.0

    if len(df) < 3:
        cov_matrix = np.identity(4)
    else:
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data)
        cov_matrix = np.cov(data_standardized, rowvar=False)
        cov_matrix += np.eye(4) * 1e-6
    cov_indices = np.triu_indices(4)
    cov_elements = cov_matrix[cov_indices]

    ellipse_params = np.array([a, b])

    return np.concatenate([stats, hist, [point_count, fraction_inside], cov_elements, ellipse_params])

LATTICE_CONTEXT_FEATURES = ("angle", "tilt")


def nn_observation_size(n_bins=5):
    """Return the fixed observation width for a configured x-y histogram."""
    return 16 + int(n_bins) ** 2 + 1 + 1 + 10 + 4 + len(
        LATTICE_CONTEXT_FEATURES
    )


def lattice_context_for_watch_point(watch_point, graph):
    """Return ``(angle, tilt)`` for the magnet following ``watch_point``.

    Automatically generated WATCH elements are placed immediately before the
    controllable magnet they describe.  MAXAMP and other zero-length metadata
    are skipped for robustness, but the search stops at the next WATCH.

    ANGLE is zero for a quadrupole.  For a dipole, its sign is preserved from
    the lattice and inverted for a reversed occurrence.  TILT is returned in
    radians exactly as specified by the lattice, defaulting to zero.
    """
    # Import locally to keep the lattice parser independent from the action
    # registry while sharing its complete Elegant magnet-family vocabulary.
    from .Utils_magnet_actions import magnet_family, normalize_element_type

    found_watch = False
    for node in graph:
        node_type = normalize_element_type(node.get("type", ""))
        if not found_watch:
            if node_type == "WATCH" and node.get("name") == watch_point:
                found_watch = True
            continue

        if node_type == "WATCH":
            break
        family = magnet_family(node_type)
        if family is None:
            continue

        attributes = node.get("attributes") or {}
        try:
            tilt = float(attributes.get("TILT", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid TILT for {node.get('name', '<unnamed>')}: "
                f"{attributes.get('TILT')!r}"
            ) from exc

        if family == "quadrupole":
            return 0.0, tilt

        try:
            angle = float(attributes.get("ANGLE", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid ANGLE for {node.get('name', '<unnamed>')}: "
                f"{attributes.get('ANGLE')!r}"
            ) from exc
        if node.get("reverse", False):
            angle = -angle
        return angle, tilt

    return 0.0, 0.0


def create_nn_representation_new(
    df,
    a_prev,
    b_prev,
    a_next,
    b_next,
    n_bins=5,
    initialNumParticles=0,
    angle=0.0,
    tilt=0.0,
):
    """
    Create a fixed-size feature vector for neural network input from particle data.
    Combines robust statistical summaries, 2D histogram of (x, y), point count,
    fraction of points inside an elliptical region, covariance matrix for x, y, xp, yp,
    and ellipse parameters. Handles empty or small datasets and is robust to outliers.

    Parameters:
    - df: pandas DataFrame with columns ['x', 'y', 'xp', 'yp']
    - a: semi-major axis of the ellipse (x-axis)
    - b: semi-minor axis of the ellipse (y-axis)
    - n_bins: number of bins per dimension for 2D histogram (default: 5)

    Returns:
    - 1D numpy array with:
        - Median, IQR, 10th, 90th percentiles for x, y, xp, yp (4 * 4 = 16)
        - 2D histogram bin counts for (x, y) (n_bins * n_bins = 25 for n_bins=5)
        - Number of points (1)
        - Fraction of points inside ellipse (1)
        - Covariance matrix upper triangle for x, y, xp, yp (10 elements)
        - Ellipse parameters a_previous, b_previous and a_next, b_next (4)
        - Upcoming magnet signed bend angle and tilt in radians (2)
        Total size for n_bins=5: 16 + 25 + 1 + 1 + 10 + 4 + 2 = 59
    """
    output_size = nn_observation_size(n_bins)
    if df.empty:
        observation = np.zeros(output_size)
        observation[-2:] = [angle, tilt]
        return observation

    features = ['x', 'y', 'xp', 'yp']
    data = df[features].to_numpy()

    stats = []
    for i in range(data.shape[1]):
        col = data[:, i]
        if len(col) > 0:
            median = np.median(col)
            p10, p90 = np.percentile(col, [10, 90])
            iqr = np.percentile(col, 75) - np.percentile(col, 25) if len(col) > 1 else 0.0
        else:
            median, iqr, p10, p90 = 0.0, 0.0, 0.0, 0.0
        stats.extend([median, iqr, p10, p90])

    if len(df) > 0:
        x_range = np.percentile(df['x'], [1, 99]) if df['x'].std() > 0 else [-1, 1]
        y_range = np.percentile(df['y'], [1, 99]) if df['y'].std() > 0 else [-1, 1]
        hist, xedges, yedges = np.histogram2d(
            df['x'], df['y'], bins=n_bins, range=[x_range, y_range]
        )
        hist = hist.flatten() / len(df) if len(df) > 0 else np.zeros(n_bins**2)
    else:
        hist = np.zeros(n_bins**2)

    if initialNumParticles <= 0:
        raise ValueError("initialNumParticles must be positive for a non-empty beam")
    point_count = len(df) / initialNumParticles
    fraction_inside = points_in_region(df, a_prev, b_prev) / 100 if not df.empty else 0.0

    if len(df) < 3:
        cov_matrix = np.identity(4)
    else:
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data)
        cov_matrix = np.cov(data_standardized, rowvar=False)
        cov_matrix += np.eye(4) * 1e-6
    cov_indices = np.triu_indices(4)
    cov_elements = cov_matrix[cov_indices]

    ellipse_params = np.array([a_prev, b_prev, a_next, b_next])
    lattice_context = np.array([angle, tilt], dtype=float)

    return np.concatenate(
        [
            stats,
            hist,
            [point_count, fraction_inside],
            cov_elements,
            ellipse_params,
            lattice_context,
        ]
    )


def points_in_region(df, a, b):
    """
    Calculate the percentage of points within an elliptical region defined by
    semi-major axis a (x-axis) and semi-minor axis b (y-axis).
    Returns the percentage (0 to 100) or 0 if DataFrame is empty.
    """
    if df.empty:
        return 0.0

    in_region = (df['x']**2 / a**2 + df['y']**2 / b**2) <= 1
    percentage = (in_region.sum() / len(df)) * 100
    return percentage

def process_particle_data(file_path, watch_point, graph, watch_points, n_bins=5, initialNumParticles=0):
    """
    Process a particle data CSV file and return the neural network representation
    and the percentage of points within the specified elliptical region defined by
    the MAXAMP element preceding the watch point.

    Parameters:
    - file_path (str): Path to CSV file.
    - watch_point (str): Name of the watch point (e.g., 'WQ1L0_1').
    - graph (list): List of dictionaries with graph elements.
    - watch_points (list): List of watch point names in chronological order.
    - n_bins (int): Number of bins per dimension for 2D histogram (default: 5).

    Returns:
    - dict or None: Dictionary with neural network representation and percentage in region,
      or None if processing fails or no MAXAMP is found.
    """
    try:
        # Find the corresponding MAXAMP element
        #maxamp = find_maxamp_for_watch_points(watch_point, graph, watch_points)
        prev_maxamp ,next_maxamp = find_next_maxamp_for_watch_points(watch_point, graph, watch_points)
        
        if prev_maxamp is None or 'attributes' not in prev_maxamp or 'X_MAX' not in prev_maxamp['attributes'] or 'Y_MAX' not in prev_maxamp['attributes']:
            print(f"Error: No valid MAXAMP found for watch point {watch_point}")
            return None
        if next_maxamp is None or 'attributes' not in next_maxamp or 'X_MAX' not in next_maxamp['attributes'] or 'Y_MAX' not in next_maxamp['attributes']:
            print(f"Error: No valid MAXAMP found for watch point {watch_point}")
            return None

        # Extract X_MAX and Y_MAX as a and b
        a_prev = float(prev_maxamp['attributes']['X_MAX'])
        b_prev = float(prev_maxamp['attributes']['Y_MAX'])

         # Extract X_MAX and Y_MAX as a and b
        a_next = float(next_maxamp['attributes']['X_MAX'])
        b_next = float(next_maxamp['attributes']['Y_MAX'])

        # Read CSV file
        df = pd.read_csv(file_path)

        # Validate required columns
        required_columns = ['x', 'y', 'xp', 'yp']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV file must contain columns: x, y, xp, yp")

        angle, tilt = lattice_context_for_watch_point(watch_point, graph)

        # Create neural network representation.  The last two values describe
        # the upcoming controllable magnet in lattice coordinates.
        nn_representation = create_nn_representation_new(
            df,
            a_prev,
            b_prev,
            a_next,
            b_next,
            n_bins,
            initialNumParticles=initialNumParticles,
            angle=angle,
            tilt=tilt,
        )

        # Calculate percentage of points in elliptical region
        percentage_in_region = points_in_region(df, a_prev, b_prev)

        return {
            'nn_representation': nn_representation,
            'percentage_in_region': percentage_in_region,
            'a_prev': a_prev,
            'b_prev': b_prev,
            'a_next': a_next,
            'b_next': b_next,
            'angle': angle,
            'tilt': tilt,
        }

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

# ---------------------------------------------------------------------------
# Safer lattice preprocessing, graph, and validation overrides.
# Definitions below replace only the corresponding lattice-related functions.
# ---------------------------------------------------------------------------

import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


CONTROL_PARAMETERS: Tuple[str, ...] = ("K1", "HKICK", "VKICK", "FSE")

# Elegant accepts a number of short aliases.  Graph consumers should see one
# stable spelling while the lattice text itself remains unchanged.
ELEMENT_TYPE_ALIASES: Dict[str, str] = {
    "DRIF": "DRIFT",
    "EDRIFT": "DRIFT",
    "CSRDRIFT": "DRIFT",
    "SBEN": "SBEND",
    "RBEN": "SBEND",
    "CSBEND": "SBEND",
    "KQUAD": "QUAD",
    "RFCA": "RFCA",
    "MONITOR": "MONI",
}


class LatticeParseError(ValueError):
    """Raised when an Elegant lattice cannot be expanded unambiguously."""


class LatticeValidationError(RuntimeError):
    """Raised when structural or Elegant validation fails."""


@dataclass(frozen=True)
class _Statement:
    raw: str
    code: str
    name: Optional[str] = None
    rhs: Optional[str] = None


@dataclass(frozen=True)
class _Definition:
    name: str
    rhs: str
    element_type: str


@dataclass(frozen=True)
class _LineItem:
    name: str
    repeat: int = 1
    reverse: bool = False
    raw: str = ""


@dataclass(frozen=True)
class _Occurrence:
    name: str
    definition: _Definition
    reverse: bool = False


_NAME_PATTERN = r"[A-Za-z_][A-Za-z0-9_.$]*"
_DEFINITION_RE = re.compile(
    rf"^\s*({_NAME_PATTERN})\s*:\s*(.*?)\s*$", re.DOTALL
)
_LINE_RE = re.compile(r"^\s*LINE\s*=\s*\((.*)\)\s*$", re.IGNORECASE | re.DOTALL)
_CONTROL_ASSIGNMENT_RE = re.compile(
    r"\b(K1|HKICK|VKICK|FSE)(\s*=\s*)"
    r"(\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|[^,\s&!]+)",
    re.IGNORECASE,
)


def _split_code_comment(line: str) -> Tuple[str, str]:
    """Split an Elegant physical line at the first unquoted ``!``."""
    quote: Optional[str] = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote:
            escaped = True
            continue
        if char in ("'", '"'):
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if char == "!" and quote is None:
            return line[:index], line[index:]
    return line, ""


def _iter_statements(content: str) -> Iterable[_Statement]:
    """Yield logical Elegant statements while retaining their original text."""
    lines = content.splitlines(keepends=True)
    index = 0
    while index < len(lines):
        first_raw = lines[index]
        first_body = first_raw.rstrip("\r\n")
        first_code, _ = _split_code_comment(first_body)

        if not first_code.strip():
            yield _Statement(raw=first_raw, code="")
            index += 1
            continue

        raw_parts: List[str] = []
        code_parts: List[str] = []
        while index < len(lines):
            raw = lines[index]
            body = raw.rstrip("\r\n")
            code, _ = _split_code_comment(body)
            stripped = code.rstrip()
            continued = stripped.endswith("&")
            if continued:
                stripped = stripped[:-1]
            raw_parts.append(raw)
            code_parts.append(stripped.strip())
            index += 1
            if not continued:
                break

        raw_text = "".join(raw_parts)
        logical_code = " ".join(part for part in code_parts if part).strip()
        match = _DEFINITION_RE.match(logical_code)
        if match:
            yield _Statement(
                raw=raw_text,
                code=logical_code,
                name=match.group(1),
                rhs=match.group(2),
            )
        else:
            yield _Statement(raw=raw_text, code=logical_code)


def _split_top_level(value: str, delimiter: str = ",") -> List[str]:
    """Split outside quotes and nested parentheses/brackets/braces."""
    parts: List[str] = []
    start = 0
    quote: Optional[str] = None
    escaped = False
    depth = 0
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = set(pairs.values())

    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote:
            escaped = True
            continue
        if char in ("'", '"'):
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if quote:
            continue
        if char in pairs:
            depth += 1
        elif char in closing:
            depth -= 1
            if depth < 0:
                raise LatticeParseError(f"Unbalanced delimiter in: {value}")
        elif char == delimiter and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1

    if quote or depth != 0:
        raise LatticeParseError(f"Unbalanced quotes or delimiters in: {value}")
    parts.append(value[start:].strip())
    return [part for part in parts if part]


def _normalize_element_type(element_type: str) -> str:
    value = element_type.strip().upper()
    return ELEMENT_TYPE_ALIASES.get(value, value)


def _definition_type_and_parameters(rhs: str) -> Tuple[str, Dict[str, str]]:
    if _LINE_RE.match(rhs):
        return "LINE", {}
    parts = _split_top_level(rhs)
    if not parts:
        return "UNKNOWN", {}
    element_type = _normalize_element_type(parts[0].split()[0])
    parameters: Dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parameters[key.strip().upper()] = value.strip()
    return element_type, parameters


def _definition_table(
    content: str, *, reject_duplicates: bool = True
) -> Tuple[List[_Statement], Dict[str, _Definition]]:
    statements = list(_iter_statements(content))
    definitions: Dict[str, _Definition] = {}
    for statement in statements:
        if not statement.name or statement.rhs is None:
            continue
        element_type, _ = _definition_type_and_parameters(statement.rhs)
        key = statement.name.casefold()
        if reject_duplicates and key in definitions:
            raise LatticeParseError(
                f"Multiple active definitions of element {statement.name}"
            )
        definitions[key] = _Definition(
            name=statement.name,
            rhs=statement.rhs,
            element_type=element_type,
        )
    return statements, definitions


def _parse_line_items(rhs: str) -> List[_LineItem]:
    match = _LINE_RE.match(rhs)
    if not match:
        raise LatticeParseError(f"Not a LINE definition: {rhs}")

    items: List[_LineItem] = []
    for raw_item in _split_top_level(match.group(1)):
        token = raw_item.strip()
        repeat = 1
        repeat_match = re.match(r"^(\d+)\s*\*\s*(.+)$", token)
        if repeat_match:
            repeat = int(repeat_match.group(1))
            token = repeat_match.group(2).strip()
        if repeat < 1:
            raise LatticeParseError(f"Invalid repetition count in {raw_item}")

        reverse = False
        if token.startswith(("-", "+")):
            reverse = token[0] == "-"
            token = token[1:].strip()
        if not re.fullmatch(_NAME_PATTERN, token):
            raise LatticeParseError(
                f"Unsupported LINE item {raw_item!r}; expected [N*][-]NAME"
            )
        items.append(_LineItem(token, repeat, reverse, raw_item))
    return items


def _reverse_occurrences(occurrences: Sequence[_Occurrence]) -> List[_Occurrence]:
    return [
        replace(occurrence, reverse=not occurrence.reverse)
        for occurrence in reversed(occurrences)
    ]


def _expand_from_definitions(
    definitions: Mapping[str, _Definition], beamline_name: str
) -> List[_Occurrence]:
    def expand(name: str, stack: Tuple[str, ...]) -> List[_Occurrence]:
        key = name.casefold()
        if key not in definitions:
            path = " -> ".join(stack + (name,))
            raise LatticeParseError(f"Undefined lattice item {name!r} in {path}")
        if key in stack:
            path = " -> ".join(stack + (key,))
            raise LatticeParseError(f"Recursive LINE definition detected: {path}")

        definition = definitions[key]
        if definition.element_type != "LINE":
            return [_Occurrence(definition.name, definition)]

        expanded: List[_Occurrence] = []
        for item in _parse_line_items(definition.rhs):
            item_occurrences = expand(item.name, stack + (key,))
            if item.reverse:
                item_occurrences = _reverse_occurrences(item_occurrences)
            for _ in range(item.repeat):
                expanded.extend(item_occurrences)
        return expanded

    return expand(beamline_name, ())


def _inline_comments(raw: str) -> List[str]:
    comments: List[str] = []
    for line in raw.splitlines():
        _, comment = _split_code_comment(line)
        if comment:
            comments.append(comment)
    return comments


def _render_line_statement(name: str, tokens: Sequence[str], old_raw: str) -> str:
    comments = _inline_comments(old_raw)
    suffix = ""
    if comments:
        suffix = "  " + "  ".join(comments)
    newline = "\n" if old_raw.endswith(("\n", "\r")) else ""
    return f"{name}: LINE=({', '.join(tokens)}){suffix}{newline}"


def _replace_statement(
    statements: Sequence[_Statement], target_name: str, replacement: str,
    *, prefix: str = ""
) -> str:
    output: List[str] = []
    replaced = False
    for statement in statements:
        if statement.name and statement.name.casefold() == target_name.casefold():
            if replaced:
                raise LatticeParseError(f"Multiple definitions of {target_name}")
            output.append(prefix)
            output.append(replacement)
            replaced = True
        else:
            output.append(statement.raw)
    if not replaced:
        raise LatticeParseError(f"Beamline {target_name!r} was not found")
    return "".join(output)


def _control_keys(definition: _Definition) -> Tuple[str, ...]:
    _, parameters = _definition_type_and_parameters(definition.rhs)
    return tuple(key for key in CONTROL_PARAMETERS if key in parameters)


def _is_controllable(definition: _Definition) -> bool:
    return bool(_control_keys(definition))


def remove_watch_points(file_path, backup_path="unchanged_input.lte"):
    """Remove active WATCH definitions/references without touching comments.

    The caller in ``Elegant.py`` invokes this on a seed-specific working copy,
    so the shared source lattice is not modified.  ``backup_path`` receives an
    exact byte-for-byte copy of the working input before modification.
    """
    source = Path(file_path)
    content = source.read_text()
    Path(backup_path).write_text(content)

    statements, definitions = _definition_table(content)
    watch_names = {
        definition.name.casefold()
        for definition in definitions.values()
        if definition.element_type == "WATCH"
    }
    if not watch_names:
        return

    output: List[str] = []
    for statement in statements:
        if statement.name and statement.name.casefold() in watch_names:
            continue
        if statement.name and statement.rhs and _LINE_RE.match(statement.rhs):
            items = _parse_line_items(statement.rhs)
            filtered = [item for item in items if item.name.casefold() not in watch_names]
            if len(filtered) != len(items):
                tokens = [item.raw.strip() for item in filtered]
                output.append(_render_line_statement(statement.name, tokens, statement.raw))
                continue
        output.append(statement.raw)
    source.write_text("".join(output))


def parse_lattice_file(file_path):
    """Return every active element definition; never activate comments."""
    content = Path(file_path).read_text()
    _, definitions = _definition_table(content)
    elements: Dict[str, Dict[str, object]] = {}
    category = {
        "WATCH": "watch point",
        "QUAD": "quadrupole magnet",
        "SBEND": "dipole magnet",
        "DRIFT": "drift",
        "MAXAMP": "aperture",
    }
    for definition in definitions.values():
        element_type, parameters = _definition_type_and_parameters(definition.rhs)
        elements[definition.name] = {
            "type": category.get(element_type, element_type.lower()),
            "element_type": element_type,
            "parameters": parameters,
            "raw_definition": definition.rhs,
        }
    return elements


def _qualify_reused_direct_controllables(content: str) -> Tuple[str, set[str]]:
    """Qualify repeated controllable references across active LINE definitions.

    This pass considers only elements that already expose K1, HKICK, VKICK, or
    FSE.  Passive elements are never cloned.  Counting every uncommented LINE
    definition means, for example, that Q1L0 in PInject and Q1L0 in PBInject
    become Q1L0_1 and Q1L0_2 respectively.  Later, the selected beamline is
    expanded and WATCH elements are added only to the occurrences it reaches.
    """
    statements, definitions = _definition_table(content)
    direct_counts: Counter[str] = Counter()

    for statement in statements:
        if not statement.name or not statement.rhs or not _LINE_RE.match(statement.rhs):
            continue
        for item in _parse_line_items(statement.rhs):
            definition = definitions.get(item.name.casefold())
            if definition is not None and _is_controllable(definition):
                direct_counts[item.name.casefold()] += item.repeat

    repeated_names = {
        name for name, count in direct_counts.items() if count > 1
    }
    if not repeated_names:
        return content, set()

    existing_names = set(definitions)
    indices: Counter[str] = Counter()
    generated_names: set[str] = set()
    generated_definitions: List[str] = []
    output: List[str] = []
    first_line_output_index: Optional[int] = None

    for statement in statements:
        if not statement.name or not statement.rhs or not _LINE_RE.match(statement.rhs):
            output.append(statement.raw)
            continue

        if first_line_output_index is None:
            first_line_output_index = len(output)

        rewritten_tokens: List[str] = []
        changed = False
        for item in _parse_line_items(statement.rhs):
            key = item.name.casefold()
            definition = definitions.get(key)
            if (
                key not in repeated_names
                or definition is None
                or not _is_controllable(definition)
            ):
                rewritten_tokens.append(item.raw.strip())
                continue

            changed = True
            for _ in range(item.repeat):
                indices[key] += 1
                occurrence_name = f"{definition.name}_{indices[key]}"
                occurrence_key = occurrence_name.casefold()
                if occurrence_key in existing_names:
                    raise LatticeParseError(
                        f"Cannot create occurrence name {occurrence_name}: it already exists"
                    )
                existing_names.add(occurrence_key)
                generated_names.add(occurrence_key)
                generated_definitions.append(
                    f"{occurrence_name}: {definition.rhs}\n"
                )
                prefix = "-" if item.reverse else ""
                rewritten_tokens.append(f"{prefix}{occurrence_name}")

        if changed:
            output.append(
                _render_line_statement(
                    statement.name, rewritten_tokens, statement.raw
                )
            )
        else:
            output.append(statement.raw)

    generated_block = (
        "! --- generated controllable occurrence definitions: begin ---\n"
        + "".join(generated_definitions)
        + "! --- generated controllable occurrence definitions: end ---\n"
    )
    insertion_index = (
        first_line_output_index
        if first_line_output_index is not None
        else len(output)
    )
    output.insert(insertion_index, generated_block)
    return "".join(output), generated_names


def add_watch_points(
    input_file,
    output_file,
    results_path,
    beamline_name="machine",
    rename_all_repeated=True,
):
    """Flatten one beamline and insert occurrence-specific WATCH elements.

    Only repeated *controllable* elements are cloned as ``NAME_1``,
    ``NAME_2``, ... .  Passive elements, apertures, drifts, monitors, and RF
    cavities may appear repeatedly without being redefined.  Direct references
    are counted across uncommented LINE definitions; remaining repetitions
    introduced by nested/repeated LINE expansion are qualified in the selected
    beamline.  The legacy ``rename_all_repeated`` argument is retained for
    caller compatibility; when true it means "rename repeated controllables".

    WATCH elements are generated only for occurrences that already contain at
    least one of K1, HKICK, VKICK, or FSE.  Missing control parameters are not
    invented.
    """
    content = Path(input_file).read_text()
    globally_qualified_names: set[str] = set()
    if rename_all_repeated:
        content, globally_qualified_names = _qualify_reused_direct_controllables(
            content
        )
    statements, definitions = _definition_table(content)
    occurrences = [
        occurrence
        for occurrence in _expand_from_definitions(definitions, beamline_name)
        if occurrence.definition.element_type != "WATCH"
    ]
    controllable_counts = Counter(
        occurrence.name.casefold()
        for occurrence in occurrences
        if _is_controllable(occurrence.definition)
    )
    indices: Counter[str] = Counter()
    existing_names = set(definitions)

    generated_element_definitions: List[str] = []
    generated_watch_definitions: List[str] = []
    flattened_tokens: List[str] = []

    results_prefix = str(results_path)
    if results_prefix and not results_prefix.endswith(("/", "\\")):
        results_prefix += "/"

    for occurrence in occurrences:
        key = occurrence.name.casefold()
        indices[key] += 1
        occurrence_number = indices[key]
        controllable = _is_controllable(occurrence.definition)
        repeated_controllable = controllable_counts[key] > 1

        active_name = occurrence.name
        if repeated_controllable and rename_all_repeated:
            active_name = f"{occurrence.name}_{occurrence_number}"
            active_key = active_name.casefold()
            if active_key in existing_names:
                raise LatticeParseError(
                    f"Cannot create occurrence name {active_name}: it already exists"
                )
            existing_names.add(active_key)
            generated_element_definitions.append(
                f"{active_name}: {occurrence.definition.rhs}\n"
            )

        if controllable:
            occurrence_is_qualified = (
                active_name != occurrence.name
                or key in globally_qualified_names
            )
            watch_name = (
                f"W{active_name}"
                if occurrence_is_qualified
                else f"W{occurrence.name}_{occurrence_number}"
            )
            watch_key = watch_name.casefold()
            if watch_key in existing_names:
                raise LatticeParseError(
                    f"Cannot create WATCH {watch_name}: it already exists"
                )
            existing_names.add(watch_key)
            generated_watch_definitions.append(
                f'{watch_name}: WATCH, filename="{results_prefix}{watch_name}.sdds", '
                "mode=coord\n"
            )
            flattened_tokens.append(watch_name)

        token = f"-{active_name}" if occurrence.reverse else active_name
        flattened_tokens.append(token)

    generated_block_parts: List[str] = [
        "! --- generated by Utils.add_watch_points: begin ---\n"
    ]
    generated_block_parts.extend(generated_element_definitions)
    generated_block_parts.extend(generated_watch_definitions)
    generated_block_parts.append(
        "! --- generated by Utils.add_watch_points: end ---\n"
    )
    generated_block = "".join(generated_block_parts)

    target_statement = next(
        (
            statement
            for statement in statements
            if statement.name and statement.name.casefold() == beamline_name.casefold()
        ),
        None,
    )
    if target_statement is None:
        raise LatticeParseError(f"Beamline {beamline_name!r} was not found")
    new_line = _render_line_statement(
        target_statement.name, flattened_tokens, target_statement.raw
    )
    output_content = _replace_statement(
        statements, beamline_name, new_line, prefix=generated_block
    )
    Path(output_file).write_text(output_content)
    validate_lattice_structure(output_file, beamline_name)


def add_final_watch_point(
    input_file, output_file, beamline_name, results_path
):
    """Append ``final_WP`` to the selected beamline exactly once."""
    content = Path(input_file).read_text()
    statements, definitions = _definition_table(content)
    target = definitions.get(str(beamline_name).casefold())
    if target is None or target.element_type != "LINE":
        raise LatticeParseError(f"LINE {beamline_name!r} was not found")

    items = _parse_line_items(target.rhs)
    tokens = [item.raw.strip() for item in items]
    if not any(item.name.casefold() == "final_wp" for item in items):
        tokens.append("final_WP")

    final_definition = definitions.get("final_wp")
    prefix = ""
    if final_definition is None:
        results_prefix = str(results_path)
        if results_prefix and not results_prefix.endswith(("/", "\\")):
            results_prefix += "/"
        prefix = (
            f'final_WP: WATCH, filename="{results_prefix}final_WP.sdds", '
            "mode=coord\n"
        )
    elif final_definition.element_type != "WATCH":
        raise LatticeParseError("final_WP exists but is not a WATCH element")

    target_statement = next(
        statement
        for statement in statements
        if statement.name and statement.name.casefold() == str(beamline_name).casefold()
    )
    replacement = _render_line_statement(
        target_statement.name, tokens, target_statement.raw
    )
    output_content = _replace_statement(
        statements, str(beamline_name), replacement, prefix=prefix
    )
    Path(output_file).write_text(output_content)
    validate_lattice_structure(output_file, str(beamline_name))


def _rewrite_control_values(
    raw: str, element_name: str, *, placeholders: bool
) -> Tuple[str, Tuple[str, ...]]:
    """Rewrite control values only in code, leaving comments byte-for-byte."""
    found: List[str] = []

    def replace_assignment(match: re.Match[str]) -> str:
        key = match.group(1).upper()
        found.append(key)
        value = f"{{variables['{element_name}{key}']}}" if placeholders else "0"
        return f"{match.group(1)}{match.group(2)}{value}"

    rewritten_lines: List[str] = []
    for line in raw.splitlines(keepends=True):
        newline = ""
        body = line
        if line.endswith("\r\n"):
            body, newline = line[:-2], "\r\n"
        elif line.endswith(("\n", "\r")):
            body, newline = line[:-1], line[-1]
        code, comment = _split_code_comment(body)
        rewritten_code = _CONTROL_ASSIGNMENT_RE.sub(replace_assignment, code)
        rewritten_lines.append(rewritten_code + comment + newline)
    ordered = tuple(key for key in CONTROL_PARAMETERS if key in found)
    return "".join(rewritten_lines), ordered


def change_initial_content(file_path, output_file, beamline_name="machine"):
    """Create an agent template and a valid zero-control reset lattice.

    ``output_file`` receives a lattice where only existing active K1, HKICK,
    VKICK, and FSE values are set to zero.  The returned string has placeholders
    in those same positions.  Replacing the placeholders with agent actions
    therefore creates the simulation lattice without reconstructing or dropping
    any other element parameter.
    """
    content = Path(file_path).read_text()
    statements, definitions = _definition_table(content)
    active_occurrences = _expand_from_definitions(definitions, beamline_name)
    active_names: List[str] = []
    seen = set()
    for occurrence in active_occurrences:
        key = occurrence.name.casefold()
        if occurrence.definition.element_type == "WATCH" or key in seen:
            continue
        seen.add(key)
        if _is_controllable(occurrence.definition):
            active_names.append(occurrence.name)
    active_keys = {name.casefold() for name in active_names}

    template_parts: List[str] = []
    reset_parts: List[str] = []
    controls_by_name: Dict[str, Tuple[str, ...]] = {}
    for statement in statements:
        if (
            statement.name
            and statement.rhs is not None
            and statement.name.casefold() in active_keys
        ):
            template_raw, template_keys = _rewrite_control_values(
                statement.raw, statement.name, placeholders=True
            )
            reset_raw, reset_keys = _rewrite_control_values(
                statement.raw, statement.name, placeholders=False
            )
            if template_keys != reset_keys:
                raise LatticeValidationError(
                    f"Control rewrite mismatch for {statement.name}"
                )
            controls_by_name[statement.name.casefold()] = template_keys
            template_parts.append(template_raw)
            reset_parts.append(reset_raw)
        else:
            template_parts.append(statement.raw)
            reset_parts.append(statement.raw)

    variables: List[str] = []
    for name in active_names:
        for key in controls_by_name.get(name.casefold(), ()):
            variables.append(f"{name}{key}")

    formatted_lattice = "".join(template_parts)
    reset_lattice = "".join(reset_parts)
    Path(output_file).write_text(reset_lattice)
    validate_lattice_structure(output_file, beamline_name)
    return formatted_lattice, variables


def parse_lte_file(file_path):
    """Parse all active one- or multi-line definitions into ``name -> RHS``."""
    content = Path(file_path).read_text()
    _, definitions = _definition_table(content)
    return {definition.name: definition.rhs for definition in definitions.values()}


def expand_beamline(element_definitions, beamline_name):
    """Expand a LINE structurally, including repetition and reversal."""
    definitions: Dict[str, _Definition] = {}
    for name, rhs in element_definitions.items():
        element_type, _ = _definition_type_and_parameters(rhs)
        definitions[str(name).casefold()] = _Definition(
            str(name), str(rhs), element_type
        )
    occurrences = _expand_from_definitions(definitions, beamline_name)
    return [
        (f"-{item.name}" if item.reverse else item.name, item.definition.rhs)
        for item in occurrences
    ]


def create_graph(expanded_beamline):
    """Create a lossless occurrence sequence for every Elegant element type."""
    graph: List[Dict[str, object]] = []
    for item in expanded_beamline:
        if not item:
            continue
        raw_name = str(item[0])
        reverse = raw_name.startswith("-")
        element_name = raw_name[1:] if reverse else raw_name
        definition = str(item[1]) if len(item) > 1 else ""
        if definition:
            element_type, attributes = _definition_type_and_parameters(definition)
        else:
            element_type, attributes = "UNKNOWN", {}
        graph.append(
            {
                "name": element_name,
                "type": element_type,
                "attributes": attributes,
                "controllable": any(
                    key in attributes for key in CONTROL_PARAMETERS
                ),
                "controllable_parameters": [
                    key for key in CONTROL_PARAMETERS if key in attributes
                ],
                "reverse": reverse,
                "raw_definition": definition,
            }
        )
    return graph


def process_lte_file_to_graph(file_path, beamline_name):
    definitions = parse_lte_file(file_path)
    return create_graph(expand_beamline(definitions, beamline_name))


def create_feature_matrix(graph):
    """Create a safe fixed-width matrix for the normalized graph vocabulary."""
    element_types = [
        "CHARGE",
        "WATCH",
        "MAXAMP",
        "RFCA",
        "QUAD",
        "SBEND",
        "DRIFT",
        "SEXT",
        "MARK",
        "MARKER",
        "MONI",
        "KICKER",
        "HKICK",
        "VKICK",
        "UNKNOWN",
        "OTHER",
    ]
    attribute_names = [
        "L",
        "K1",
        "K2",
        "K3",
        "HKICK",
        "VKICK",
        "ANGLE",
        "FSE",
        "TILT",
        "X_MAX",
        "Y_MAX",
        "ELLIPTICAL",
        "VOLT",
        "FREQ",
        "PHASE",
    ]
    encoder = OneHotEncoder(categories=[element_types], handle_unknown="ignore")
    encoder.fit(np.asarray(element_types).reshape(-1, 1))

    rows: List[np.ndarray] = []
    for node in graph:
        raw_type = _normalize_element_type(str(node.get("type", "UNKNOWN")))
        encoded_type = raw_type if raw_type in element_types else "OTHER"
        type_vector = encoder.transform([[encoded_type]]).toarray().ravel()
        attributes = node.get("attributes") or {}
        numeric: List[float] = []
        for name in attribute_names:
            try:
                numeric.append(float(attributes.get(name, 0)))
            except (TypeError, ValueError):
                numeric.append(0.0)
        rows.append(np.concatenate((type_vector, np.asarray(numeric))))

    width = len(element_types) + len(attribute_names)
    if not rows:
        return np.empty((0, width), dtype=float)
    matrix = np.asarray(rows, dtype=float)
    matrix[:, len(element_types) :] = MinMaxScaler().fit_transform(
        matrix[:, len(element_types) :]
    )
    return matrix


def validate_lattice_structure(file_path, beamline_name="machine"):
    """Fail fast on duplicates, undefined items, or misplaced generated WATCHes."""
    content = Path(file_path).read_text()
    _, definitions = _definition_table(content, reject_duplicates=True)
    occurrences = _expand_from_definitions(definitions, beamline_name)
    graph = create_graph(
        [
            (
                f"-{occurrence.name}" if occurrence.reverse else occurrence.name,
                occurrence.definition.rhs,
            )
            for occurrence in occurrences
        ]
    )

    for index, occurrence in enumerate(occurrences):
        if not _is_controllable(occurrence.definition):
            continue
        if index == 0 or occurrences[index - 1].definition.element_type != "WATCH":
            raise LatticeValidationError(
                f"Controllable element {occurrence.name} does not have a WATCH immediately before it"
            )

    return {
        "beamline": beamline_name,
        "occurrences": len(occurrences),
        "watch_points": sum(
            occurrence.definition.element_type == "WATCH"
            for occurrence in occurrences
        ),
        "types": dict(Counter(str(node["type"]) for node in graph)),
        "graph": graph,
    }


def validate_with_elegant(
    lattice_file,
    beamline_name="machine",
    elegant_executable="elegant",
    rpn_defns=None,
    p_central_mev=10.0,
    timeout=30,
):
    """Ask Elegant itself to parse the generated lattice before training.

    This performs only ``run_setup``.  It validates definitions and beamline
    expansion without attempting tracking, Twiss matching, or particle I/O.
    The caller should pass the same Elegant executable/RPN definitions used by
    the training environment.
    """
    lattice_path = Path(lattice_file).resolve()
    executable = str(elegant_executable)
    if not (Path(executable).is_file() or shutil.which(executable)):
        raise FileNotFoundError(f"Elegant executable not found: {executable}")

    with tempfile.TemporaryDirectory(prefix="elegant_lattice_validation_") as temp_dir:
        command_file = Path(temp_dir) / "validate.ele"
        command_file.write_text(
            "&run_setup\n"
            f'  lattice = "{lattice_path}",\n'
            f"  use_beamline = {beamline_name},\n"
            f"  p_central_mev = {float(p_central_mev)}\n"
            "&end\n"
        )
        command = [executable]
        if rpn_defns:
            command.append(f"-rpnDefns={Path(rpn_defns).resolve()}")
        command.append(str(command_file))
        result = subprocess.run(
            command,
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
        if result.returncode != 0 or re.search(r"(?im)^\s*error:", combined):
            raise LatticeValidationError(
                f"Elegant rejected {lattice_path}:\n{combined.strip()}"
            )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

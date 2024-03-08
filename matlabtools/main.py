#%%
import numpy as np
import pandas as pd
import warnings
import os
import tempfile
from pathlib import Path
import scipy.io

#%%
# Set the print options to show only up to 5 elements in the array representation
np.set_printoptions(threshold=5, precision=5)

def is_iterable(obj):
    """
    Check if an object is iterable. Note that empy arrays or lists are considered iterable, so you may want to check also for the length of the object.
    """
    if isinstance(obj, dict):
        return False
    else:
        try:
            _ = obj[0]
            return True
        except IndexError: # must be an empty list or array
            return True
        except TypeError:
            return False

class AttributeDict(dict):
    """
    Convert a dictionary to an object with attributes. The advantage is that you can access the dictionary values by using either the key or the attribute name:
    Example usage:
        input_dict = {"name": "John", "age": 30, "city": "New York"}
        obj = AttributeDict(input_dict)
        print(obj.name)
        print(obj['name'])
    """
    # def hasattr(self, key):
    #     return key in self.keys()
    
    def __getattr__(self, key):
        if key in self.keys():
            return self[key]
        else: 
            # this is crucial for built-in hasttr(obj, key) to work
            raise AttributeError(f"Object {self.__class__.__name__} has no attribute {key}")
    

    def __setattr__(self, key, value):
        self[key] = value
        
    def __delattr__(self, key: str):
        self.pop(key, None)

    def __repr__(self):
        """
        Improve the representation of the object when printed.
        """
        def get_repr(value, key):

            if value is None:
                return 'None'

            # concise representation of arrays containing objects:
            if isinstance(value, np.ndarray) or isinstance(value, list) or isinstance(value, tuple): 
            
                if len(value)>0 and isinstance(value[0], dict):
                    return f"{value.shape} {type(value).__name__} ({value.dtype})"
            
            # concise representation of multidimensional numpy arrays:
            elif isinstance(value, np.ndarray) and value.size > 10 and value.ndim > 1:
                return f"{value.shape} {type(value).__name__} ({value.dtype})"
            
            
            # concise representation of pandas Dataframe objects:
            try:
                if isinstance(value, pd.DataFrame):
                    return f"{type(value).__name__} {value.shape}"
            
                if is_iterable(value):
                    if len(value)>0:
                        if isinstance(value[0], pd.DataFrame):
                            return f"{type(value)} of: {type(value[0]).__name__} (size: {len(value)})"
            
            except Exception as e:
                warnings.warn(f"Error while trying to get the representation of attribute {key} of type: {type(value).__name__}). Got error:\n{e}")
            
            
            # display only one level of nested dictionaries
            if isinstance(value, dict):
                return f"{type(value).__name__}()"
            # if isinstance(value, dict):
            #     return f"{type(value).__name__}({', '.join(f'{k}={type(v).__name__}' for k, v in value.items())})"
            
            return value

        try:
            items = [(key, get_repr(value, key)) for key, value in self.items()]
            if len(items) > 0:
                max_key_length = max(len(str(key)) for key, _ in items)
                return "\n".join([f"{str(key).ljust(max_key_length)}: {value}" for key, value in items])
                # return dict(items).__repr__()
            else:
                return dict(items).__repr__()
        
        except Exception as e:
            warnings.warn(f"Error while trying to get the representation of object {self.__class__.__name__}: {e}")
            return dict(self).__repr__()
        
class Struct(AttributeDict):
    """
    Same as AttributeDict, but extends it so that it can be saved to and loaded from a .mat file using scipy.io.savemat and scipy.io.loadmat.
    Example usage:
        my_obj = Struct()
        my_obj['data1'] = [1, 2, 3]
        my_obj['data2'] = 'example'
        my_obj.data3 = np.array([1, 10])
        my_obj.to_mat('my_object.mat')
        obj = Struct.from_mat('my_object.mat')
    """    

    try:
        sio_mat_struct = scipy.io.matlab.mat_struct
    except AttributeError:  # for older versions of scipy
        sio_mat_struct = scipy.io.matlab.mio5_params.mat_struct
    
    def to_mat(self, p):
        import scipy.io as sio
        # Create a mat file:
        sio.savemat(p, self)

    @classmethod
    def from_mat(cls, p, mat_key=None):
        import scipy.io as sio
        # Create an instance of the class
        obj = cls()
        if mat_key is not None:
            matdata = sio.loadmat(p, struct_as_record=False, squeeze_me=True)[mat_key]
        else:
            matdata = sio.loadmat(p, struct_as_record=False, squeeze_me=True)
                
        if isinstance(matdata, Struct.sio_mat_struct):
            return cls.mat_struct_to_dict(matdata)
        else:
            return cls(matdata)
        
    
    @classmethod
    def mat_struct_to_dict(cls, mat_struct):
        """
        Convert a scipy.io.matlab.mat_struct to a Python dictionary.
        """
        # Convert the MATLAB struct to a Python dictionary
        dict = {field_name: getattr(mat_struct, field_name) for field_name in mat_struct._fieldnames}
        obj = cls(dict)
        # Recursively convert the elements of the dictionary
        for key, value in obj.items():
            
            if isinstance(value, cls.sio_mat_struct):
                dict[key] = cls.mat_struct_to_dict(value)

            elif isinstance(value, np.ndarray) and len(value) > 0:
                if isinstance(value[0], cls.sio_mat_struct):
                    dict[key] = np.array([cls.mat_struct_to_dict(x) for x in value])

        return cls(dict)

    
class StructHDF5(Struct):
    """
    Same as AttributeDict, but it can be saved to and loaded from an HDF5 file.
    # Example usage:
        my_obj = AttributeDictHDF5()
        my_obj['data1'] = [1, 2, 3]
        my_obj['data2'] = 'example'
        my_obj.data3 = np.array([1, 10])
        my_obj.to_h5('my_object.h5')
        # load it to check if the two objects match:
        obj = AttributeDictHDF5.from_h5('my_object.h5')
    """

    def to_h5(self, p):
        import h5py
        # Create or open an HDF5 file
        with h5py.File(p, 'w') as f:
            # Iterate over all attributes of the class instance
            for attr_name, attr_value in self.items():
                print(attr_name, attr_value)
                f[attr_name] = attr_value # is this equivalent to f.create_dataset(attr_name, data=attr_value)?
                # f.create_dataset(attr_name, data=attr_value)

    @classmethod
    def from_h5(cls, p):
        import h5py
        # Create an instance of the class
        obj = cls()

        # Load data from the HDF5 file
        with h5py.File(p, 'r') as f:
            # Iterate over all attributes of the class instance
            for attr_name in f.keys():
                setattr(obj, attr_name, f[attr_name][()])

        return obj

def run_as_subprocess(cmd, logdir='./log', verbose=True):
    """
    Run a command as a subprocess, and prints the output (also saved to a log file).
    Example:
        run_as_subprocess('echo $SHELL')
    """

    import datetime
    import subprocess
    from pathlib import Path

    do_log = True

    if logdir is None or logdir == '' or logdir==False:
        do_log = False

    if do_log:
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        logdir = Path(logdir)

        if not logdir.exists():
            logdir.mkdir(parents=True)

        log_fname = logdir / "{}.txt".format(dt_string)
    
    print(f'Starting subprocess with command "{cmd}" \n')

    try:
        if verbose:
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)  # Capture stdout
        else:
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # Capture stdout without verbose

        if do_log:
            with open(log_fname, "w") as log_file:
                log_file.write(result.stdout)  # Write stdout to log file

        if result.returncode != 0:
            print(f"Error running {cmd}.")
            if do_log:
                print(f"A log file was saved to {log_fname.absolute()}.")
            print("The error message reads: \n{}".format(result.stdout))
            return 1
        else:
            print("cmd {} ran successfully.".format(cmd))
            print("The output reads: \n{}".format(result.stdout))
            return 0

    except subprocess.CalledProcessError as e:
        print("Error running process")
        print("Error message: ", e.output)
        return 1


def run_matlab_script(cmd, cmd_prefix='matlab -nodesktop -nosplash -r', addpath=None, logdir=None, verbose=True):
    """
    Run a matlab script as a subprocess.
    Example:
        success=run_matlab_script('pwd;exit(0);')
    """

    # matlab_cmd = '{};exit(0);'.format(cmd)
    # matlab_cmd = "try,{},exit(0);catch ex,disp(getReport(ex));exit(1);end".format(cmd)
    
    if addpath:
        full_cmd = f'cd {addpath} && {cmd_prefix} "{cmd}" && cd -'
    else:
        full_cmd = f'{cmd_prefix} "{cmd}"'
    # return full_cmd
    return run_as_subprocess(full_cmd, logdir, verbose)

def run_matlab_code(matlab_code, logdir=None, cmd_prefix='matlab -nodesktop -nosplash -r', verbose=True):
    """
    Executes given MATLAB code on-the-fly and returns the result.
    
    :param matlab_code: A string containing the MATLAB code to execute. Note that the variable `result` must be defined in the MATLAB code as this is the variable that will be returned by this function.
    :return: The result of the MATLAB execution.
    
    Example:
        # Example MATLAB code to run
        matlab_test_code = "a = 1; b = 2; result = a + b;"
        # Run the MATLAB code
        result = run_matlab_code(matlab_test_code)
        print(f"Result of the MATLAB execution is: {result}")
    """
    # Start MATLAB engine
    # eng = matlab.engine.start_matlab()
    
    # Define a unique function name for each test
    function_name = 'tempFunction'
    
    
    # Write the MATLAB function to a temporary .m file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.m') as matlab_file:
        
        matlab_file_path = Path(matlab_file.name)  # Get the path of the temporary .m file
        matlab_file_dir = Path(matlab_file_path).parent
        fname = matlab_file_path.stem # need to remove the '.m' at the end
        output_file_path = matlab_file_dir / f'{fname}.mat'
    
        wrapper_code = '\n'.join([
            f"try",
            f"addpath('{str(matlab_file_dir)}')", # so that matlab finds the file
            f"%----------------------------- <== Your MATLAB code starts here",
            f"    {matlab_code}",
            f"%----------------------------- ==> Your MATLAB code ends here",
            f"    save('{str(output_file_path)}', 'result')",
            f"    exit(0); % necessary for benign termination of python subprocess",
            f"catch ex,",
            f"    disp(getReport(ex));",
            f"    exit(1); % necessary for benign termination of python subprocess",
            f"end"
        ])
        
        # Define the content of the MATLAB function
        function_content = f"function result = {function_name}()\n{wrapper_code}\nend"
        
        
        matlab_file.write(function_content)
    
    if verbose:
        print(f'Content of the script at path {matlab_file_path}\n-------------------------------:')
        print(function_content)
        print('-------------------------------\n')
    
    # Call the MATLAB function from the temporary .m file
    result = None
    try:
        
        
        # Navigate to the directory of the temp file
        # eng.cd(matlab_file_dir)
        
        # fail_bool = None
        fail_bool = run_matlab_script(fname, addpath=str(matlab_file_dir), cmd_prefix=cmd_prefix, logdir=logdir, verbose=verbose)
        # result = None
        
        if not fail_bool:
            # Load the result from the .mat file
            result = Struct.from_mat(output_file_path, mat_key='result')
        
        else:
            raise Exception()
        

    except Exception as e:
        raise e
    finally:
        os.remove(matlab_file_path)
        pass
    
    return result


# %%

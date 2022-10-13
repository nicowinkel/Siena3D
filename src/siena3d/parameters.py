from collections import UserDict



class Parameter(object):
    """
    A container for storing a parameter value and the corresponding
    description.

    Parameters
    ----------
    value : float
        The value of the parameter that we want to store.
    description : string, optional
        The type of object that is stored.
    """

    def __init__(self, value, description=''):
        self.__value = value
        self.__description = description

    def getValue(self):
        """Obtain the value of the parameter."""
        return self.__value

    def getDescription(self):
        """Get the description of the parameter as a string."""
        return self.__description

    def setValue(self, value):
        """Change the value of the parameter."""
        self.__value = value

    def setDescription(self, description):
        """Change the description of the parameter by providing a string."""
        self.__description = description


class ParameterList(UserDict):
    """
    A container class consisting of the parameter name as the key and a
    Parameter object (consisting of value and description) as its value.

    Parameters
    ----------
    filename : string, None, optional
        If a filename is given, the parameters will be read in from the file as
        following:
        parameterName1 parameterValue1                      #description1
        parameterName2 parameterValue2, parametervalue3     #description2
        ...
    """
    def __init__(self, filename=None):
        UserDict.__init__(self)
        if filename is not None:
            infile = open(filename, 'r')
            lines = infile.readlines()
            infile.close()
            for i in range(len(lines)):
                split = lines[i].split('#')
                if len(split) == 1:
                    prepart = split[0]
                    description = ''
                elif len(split) == 2:
                    (prepart, description) = split
                elif len(split) != 0:
                    msg = 'Error while parsing the file {}. '.format(filename)
                    msg += 'Please consult the user manual for instructions ' \
                           'on the correct formatting instructions.'
                    raise ValueError(msg)
                else:
                    continue
                (parname, value) = prepart.split()
                self[parname] = Parameter(value, description.replace('\n', ''))

    def addParameter(self, Name, parameter):
        self[Name] = parameter


def load_parlist(filename):
    """Read in the parameters file.

    Parameters
    ----------
    filename : `string`
        input file name
    """
    attributes = type('', (), {})()

    parlist = ParameterList(filename)
    
    for par in parlist:
        string = parlist[par].getValue()

        # check if input value consists of multiple values
        if ',' in string:
            value = string.split(',')
            for idx, _ in enumerate(value):
                # check if float
                try:
                    vtmp = float(value[idx])
                    # check if integer
                    if (int(vtmp) == vtmp): value[idx] = int(vtmp)
                    else: value[idx] = vtmp
                except:
                    None

        # check if input is numeric
        elif string.replace('.','',1).isnumeric():
            vtmp = float(string)
            try:
                # check if integer
                if (int(vtmp) == vtmp):
                    value = int(vtmp)
                else:
                    value = vtmp
            except:
                None

        # input is neither tuple, not int or float
        else:
            try:
                value = string
            except:
                msg = 'Error while parsing the file {}. '.format(filename)
                msg += 'Please follow the formatting of the example files or ' \
                       'consult the user manual for instructions on the formatting.'
                raise ValueError(msg)
            
        setattr(attributes, par, value)

    return attributes

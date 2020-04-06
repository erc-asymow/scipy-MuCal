# Built-in/Generic Imports
import os
import sys
import time
import ROOT

# Begin code for casting

# this injection can be replaced by properly having this in a header
# included in the interpreter at framework startup
ROOT.gInterpreter.Declare('''
template <typename T>
class NodeCaster {
   public:
   static ROOT::RDF::RNode Cast(T rdf)
   {
      return ROOT::RDF::RNode(rdf);
   }
};
''')

def CastToRNode(node):
    return ROOT.NodeCaster(node.__cppname__).Cast(node)

# end code for casting

def getValueType(obj):
    
    class_name = obj.__cppname__
    open_br, close_br = class_name.find('<'), class_name.rfind('>')
    value_type = class_name[open_br+1:close_br]
    return value_type



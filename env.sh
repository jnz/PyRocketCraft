if [[ "$OSTYPE" == "darwin"* ]]; then
  export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:"$(pwd)/../acados/lib"
else
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$(pwd)/../acados/lib"
fi
export ACADOS_SOURCE_DIR="$(pwd)/../acados"
source venv_pyrocketcraft/bin/activate

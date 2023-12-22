CURRENT_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ "$OSTYPE" == "darwin"* ]]; then
  export DYLD_LIBRARY_PATH="$CURRENT_SCRIPT_DIR/acados/lib"
else
  export LD_LIBRARY_PATH="$CURRENT_SCRIPT_DIR/acados/lib"
fi
export ACADOS_SOURCE_DIR="$CURRENT_SCRIPT_DIR/acados"
source venv_pyrocketcraft/bin/activate

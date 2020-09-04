# installing python requirements with order
echo "Installing python requirements"
pypackages=$(grep -v '^#' requirements.txt)
for package in $pypackages; do
    if [ ${package:0:4} == 'git+' ]; then
        package_name=(${package//\// })
        package_name=${package_name[-1]}
    else
        package_name=$package
    fi
    available=$(pip show $package_name | wc -l)
    if [ $available -eq 0 ]; then
        pip install $package
    fi
done

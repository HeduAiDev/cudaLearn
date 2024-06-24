add_rules("mode.debug", "mode.release")

includes("3rd")
set_languages("c++17")
add_requires("mycutlass", "eigen", {system = false})
add_links('cublas')
add_cugencodes("sm_75")
for _, file in ipairs(os.files("*.cu")) do 
    local file_name_without_ext = path.filename(file):match("(.+)%..+$")
    target(file_name_without_ext)
        set_kind("binary")
        add_files(file)
        set_targetdir("dist")
        add_packages("mycutlass")
end
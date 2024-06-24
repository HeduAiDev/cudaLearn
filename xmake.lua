add_rules("mode.debug", "mode.release")

-- includes("3rd")
set_languages("c++17")
add_requires("cutlass")
-- set_toolchains("nvcc")
add_cugencodes("sm_75")
for _, file in ipairs(os.files("*.cu", "test/*.cu")) do 
    local file_name = path.filename(file)
    target(file_name)
        set_kind("binary")
        add_files(file)
        set_targetdir("dist")
end
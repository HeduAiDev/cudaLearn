package("mycutlass")
    set_kind("library", {headeronly = true})
    set_homepage("https://github.com/NVIDIA/cutlass")
    set_description("CUDA Templates for Linear Algebra Subroutines")

    add_urls("https://github.com/NVIDIA/cutlass.git")

    add_versions("v3.5.0", "ef6af8526e3ad04f9827f35ee57eec555d09447f70a0ad0cf684a2e426ccbcb6")
    add_versions("v3.4.1", "aebd4f9088bdf2fd640d65835de30788a6c7d3615532fcbdbc626ec3754becd4")
    add_versions("v3.2.0", "9637961560a9d63a6bb3f407faf457c7dbc4246d3afb54ac7dc1e014dd7f172f")
    
    set_installdir(path.join(os.projectdir(), '3rd', 'cutlass'))

    on_install(function (package)
        local source = os.dirs("cutlass*")
        if source and #source ~= 0 then
            os.cd(source[1])
        end
        os.cp("include", package:installdir())
    end)

    on_test(function (package)
        assert(package:has_cxxincludes("cutlass/cutlass.h", {configs = {languages = "c++17"}}))
    end)

package("eigen")

    set_kind("library", {headeronly = true})
    set_homepage("https://eigen.tuxfamily.org/")
    set_description("C++ template library for linear algebra")
    set_license("MPL-2.0")

    add_urls("https://gitlab.com/libeigen/eigen/-/archive/$(version)/eigen-$(version).tar.bz2",
             "https://gitlab.com/libeigen/eigen.git")
    add_versions("3.3.7", "685adf14bd8e9c015b78097c1dc22f2f01343756f196acdc76a678e1ae352e11")
    add_versions("3.3.8", "0215c6593c4ee9f1f7f28238c4e8995584ebf3b556e9dbf933d84feb98d5b9ef")
    add_versions("3.3.9", "0fa5cafe78f66d2b501b43016858070d52ba47bd9b1016b0165a7b8e04675677")
    add_versions("3.4.0", "b4c198460eba6f28d34894e3a5710998818515104d6e74e5cc331ce31e46e626")
    
    if is_plat("mingw") and is_subhost("msys") then
        add_extsources("pacman::eigen3")
    elseif is_plat("linux") then
        add_extsources("pacman::eigen", "apt::libeigen3-dev")
    elseif is_plat("macosx") then
        add_extsources("brew::eigen")
    end

    add_deps("cmake")
    add_includedirs("include")
    add_includedirs("include/eigen3")
    set_installdir(path.join(os.projectdir(), '3rd', 'Eigen'))

    on_install("macosx", "linux", "windows", "mingw", "cross", function (package)
        import("package.tools.cmake").install(package, {"-DBUILD_TESTING=OFF"})
    end)

    on_test(function (package)
        assert(package:check_cxxsnippets({test = [[
            #include <iostream>
            #include <Eigen/Dense>
            using Eigen::MatrixXd;
            void test()
            {
                MatrixXd m(2,2);
                m(0,0) = 3;
                m(1,0) = 2.5;
                m(0,1) = -1;
                m(1,1) = m(1,0) + m(0,1);
                std::cout << m << std::endl;
            }
        ]]}, {configs = {languages = "c++11"}}))
    end)
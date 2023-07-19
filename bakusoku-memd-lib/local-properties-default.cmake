#[===[
    ここでは、find_packageでの検索パスの設定をしているよ。
    この挙動を上書きしたいときは、同じディレクトリに "local-properties.cmake" を配置すると良い。
    そこでset_local_properties関数を作り、変数をPARENT_SCOPEで定義する。

    例 自分でビルドして変な場所に配置したOpenBLASをcmakeに見つけてもらえるようにする場合
    function(set_local_properties)
        set(OpenBLAS_DIR /home/Watakushi/mylib/OpenBLAS/lib/cmake/openblas PARENT_SCOPE)
    endfunction(set_local_properties)
]===]

function(set_local_properties)
    if (APPLE)
        find_program(BREW_BIN brew)
        if (BREW_BIN)
            execute_process(COMMAND ${BREW_BIN} --prefix openblas
                    OUTPUT_VARIABLE OPENBLAS_BREW_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
            set(OpenBLAS_DIR ${OPENBLAS_BREW_PREFIX}/lib/cmake/openblas PARENT_SCOPE)

            # OpenMPのインストールチェック
            execute_process(COMMAND ${BREW_BIN} --prefix libomp
                    OUTPUT_VARIABLE OpenMP_HOME
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
            set(OpenMP_C_LIB_NAMES "omp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${OpenMP_HOME}/lib/")
            set(OpenMP_CXX_FLAGS -Xpreprocessor -fopenmp -Wno-unused-command-line-argument -I${OpenMP_HOME}/include -lomp -L${OpenMP_omp_LIBRARY} CACHE STRING "" FORCE)
            set(OpenMP_C_FLAGS "-fopenmp -Wno-unused-command-line-argument -I${OpenMP_HOME}/include -lomp -L${OpenMP_omp_LIBRARY}" CACHE STRING "" FORCE)
        endif ()
    elseif (UNIX)

    else ()

    endif ()
endfunction(set_local_properties)

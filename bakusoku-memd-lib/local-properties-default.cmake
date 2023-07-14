#[===[
    ここでは、find_packageでの検索パスの設定をしているよ。
    この挙動を上書きしたいときは、同じディレクトリに "local-properties.cmake" を配置すると良い。
    そこでset_local_properties関数を作り、変数をPARENT_SCOPEで定義する。

    例 自分でビルドして変な場所に配置したOpenBLASをcmakeに見つけてもらえるようにする場合
    function(set_local_properties)
        set(OpenBLAS_DIR /home/Watakushi/mylib/OpenBLAS/lib/cmake/openblas
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
        endif ()
    elseif (UNIX)

    else ()

    endif ()
endfunction(set_local_properties)

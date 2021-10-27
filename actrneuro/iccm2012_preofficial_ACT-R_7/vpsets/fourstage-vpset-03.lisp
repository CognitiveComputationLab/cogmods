(defparameter *term-presentations* (list
'(("K" "l" "N") ("K" "l" "X") ("X" "l" "Z") ("N" "l" "X") ("K" "N" "X" "Z") 193 1 "1" 1)
'(("K" "r" "X") ("K" "l" "N") ("N" "l" "Z") ("K" "r" "Z") ("X" "K" "N" "Z") 146 0 "0" 1)
'(("K" "l" "X") ("Z" "r" "K") ("Z" "l" "N") ("X" "l" "K") ("K" "N" "X" "Z") 326 0 "3" 0)
'(("N" "r" "K") ("K" "l" "X") ("X" "l" "Z") ("K" "r" "Z") ("K" "Z" "X" "N") 268 0 "2" 0)
'(("Z" "r" "X") ("N" "r" "Z") ("N" "l" "K") ("K" "r" "Z") ("X" "N" "Z" "K") 56 1 "0" 0)
'(("N" "r" "X") ("K" "r" "X") ("K" "l" "Z") ("X" "r" "Z") ("X" "N" "Z" "K") 224 0 "1" 0)
'(("K" "r" "Z") ("Z" "l" "N") ("N" "l" "X") ("K" "l" "Z") ("Z" "K" "X" "N") 228 0 "1" 0)
'(("N" "l" "Z") ("N" "l" "K") ("K" "l" "X") ("X" "l" "Z") ("N" "K" "X" "Z") 289 1 "3" 1)
'(("X" "l" "N") ("K" "r" "N") ("K" "l" "Z") ("K" "l" "X") ("X" "K" "Z" "N") 107 0 "0" 0)
'(("K" "r" "X") ("N" "r" "X") ("N" "l" "Z") ("K" "r" "N") ("X" "N" "K" "Z") 255 1 "2" 1)
'(("K" "l" "X") ("Z" "r" "K") ("Z" "l" "N") ("Z" "r" "X") ("K" "X" "Z" "N") 205 1 "1" 1)
'(("N" "l" "X") ("X" "l" "Z") ("Z" "l" "K") ("N" "l" "Z") ("N" "X" "Z" "K") 1 1 "0" 1)
'(("N" "l" "Z") ("X" "r" "Z") ("X" "l" "K") ("N" "l" "K") ("N" "Z" "X" "K") 67 1 "0" 1)
'(("N" "r" "Z") ("Z" "l" "X") ("X" "l" "K") ("N" "l" "X") ("Z" "N" "K" "X") 196 1 "1" 0)
'(("X" "l" "K") ("Z" "r" "X") ("Z" "l" "N") ("N" "l" "K") ("X" "N" "K" "Z") 294 1 "3" 0)
'(("K" "l" "X") ("N" "r" "K") ("N" "l" "Z") ("X" "l" "K") ("K" "X" "N" "Z") 229 0 "1" 1)
'(("N" "r" "Z") ("N" "l" "X") ("X" "l" "K") ("K" "r" "N") ("Z" "X" "N" "K") 54 1 "0" 0)
'(("Z" "r" "X") ("N" "r" "X") ("N" "l" "K") ("X" "r" "Z") ("X" "N" "Z" "K") 287 0 "2" 1)
'(("Z" "l" "K") ("Z" "l" "X") ("X" "l" "N") ("N" "l" "Z") ("Z" "X" "N" "K") 305 0 "3" 1)
'(("N" "l" "Z") ("Z" "l" "X") ("X" "l" "K") ("X" "l" "N") ("N" "X" "K" "Z") 105 0 "0" 0)
'(("Z" "l" "X") ("Z" "l" "K") ("K" "l" "N") ("Z" "r" "N") ("Z" "X" "K" "N") 217 0 "1" 1)
'(("Z" "r" "K") ("X" "r" "K") ("X" "l" "N") ("X" "r" "Z") ("K" "Z" "N" "X") 208 1 "1" 0)
'(("K" "r" "X") ("X" "l" "Z") ("Z" "l" "N") ("K" "r" "Z") ("X" "N" "Z" "K") 252 1 "2" 0)
'(("N" "r" "Z") ("K" "r" "N") ("K" "l" "X") ("Z" "r" "X") ("Z" "N" "K" "X") 180 0 "0" 1)
'(("X" "r" "Z") ("Z" "l" "N") ("N" "l" "K") ("Z" "r" "X") ("Z" "X" "K" "N") 236 0 "1" 0)
'(("Z" "l" "X") ("N" "r" "X") ("N" "l" "K") ("N" "r" "Z") ("Z" "X" "N" "K") 19 1 "0" 1)
'(("K" "l" "Z") ("X" "r" "K") ("X" "l" "N") ("N" "l" "K") ("K" "N" "X" "Z") 262 0 "2" 0)
'(("X" "l" "Z") ("K" "r" "X") ("K" "l" "N") ("Z" "l" "K") ("X" "Z" "K" "N") 197 1 "1" 1)
'(("Z" "r" "X") ("Z" "l" "K") ("K" "l" "N") ("N" "l" "X") ("X" "Z" "K" "N") 162 0 "0" 1)
'(("Z" "r" "N") ("N" "l" "K") ("K" "l" "X") ("X" "l" "Z") ("N" "X" "Z" "K") 292 1 "3" 0)
'(("Z" "l" "N") ("Z" "l" "K") ("K" "l" "X") ("K" "r" "N") ("Z" "N" "K" "X") 201 1 "1" 1)
'(("K" "l" "N") ("N" "l" "Z") ("Z" "l" "X") ("K" "r" "Z") ("K" "Z" "N" "X") 117 0 "0" 0)
'(("K" "r" "N") ("Z" "r" "N") ("Z" "l" "X") ("N" "r" "K") ("N" "Z" "X" "K") 335 0 "3" 1)
'(("N" "l" "K") ("N" "l" "X") ("X" "l" "Z") ("K" "r" "X") ("N" "X" "K" "Z") 249 1 "2" 1)
'(("X" "r" "Z") ("K" "r" "X") ("K" "l" "N") ("X" "l" "N") ("Z" "K" "N" "X") 44 1 "0" 0)
'(("X" "r" "K") ("N" "r" "K") ("N" "l" "Z") ("Z" "l" "K") ("K" "X" "Z" "N") 216 0 "1" 0)
'(("N" "l" "X") ("K" "r" "N") ("K" "l" "Z") ("Z" "l" "X") ("N" "K" "Z" "X") 293 1 "3" 1)
'(("N" "l" "K") ("X" "r" "K") ("X" "l" "Z") ("K" "r" "Z") ("N" "K" "X" "Z") 147 0 "0" 1)
'(("K" "r" "N") ("N" "l" "Z") ("Z" "l" "X") ("Z" "r" "K") ("N" "K" "X" "Z") 204 1 "1" 0)
'(("N" "r" "X") ("Z" "r" "N") ("Z" "l" "K") ("K" "l" "X") ("X" "Z" "N" "K") 168 0 "0" 0)
'(("Z" "l" "K") ("X" "r" "Z") ("X" "l" "N") ("Z" "r" "K") ("Z" "K" "X" "N") 237 0 "1" 1)
'(("X" "r" "N") ("K" "r" "N") ("K" "l" "Z") ("X" "l" "N") ("N" "Z" "X" "K") 328 0 "3" 0)
'(("N" "r" "K") ("K" "l" "Z") ("Z" "l" "X") ("N" "r" "Z") ("K" "Z" "N" "X") 251 1 "2" 1)
'(("Z" "l" "K") ("Z" "l" "N") ("N" "l" "X") ("X" "l" "Z") ("Z" "K" "N" "X") 209 0 "1" 1)
'(("X" "l" "N") ("N" "l" "Z") ("Z" "l" "K") ("K" "r" "N") ("X" "N" "Z" "K") 49 1 "0" 1)
'(("Z" "r" "K") ("N" "r" "K") ("N" "l" "X") ("Z" "l" "N") ("K" "Z" "X" "N") 200 1 "1" 0)
'(("X" "r" "N") ("X" "l" "Z") ("Z" "l" "K") ("N" "l" "Z") ("N" "Z" "K" "X") 10 1 "0" 0)
'(("K" "l" "Z") ("K" "l" "N") ("N" "l" "X") ("K" "r" "X") ("K" "X" "N" "Z") 266 0 "2" 0)
))
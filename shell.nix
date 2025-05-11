{ pkgs ? import <nixpkgs> {} }:

with pkgs;

mkShell {
  buildInputs = [
    pkgs.uv
  ];

  shellHook = ''
    uv --version
    uv add scikit-learn
  '';
}
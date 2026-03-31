#!/usr/bin/env python3
"""
generate_xcodeproj.py — Generates SwiftLMChat.xcodeproj without xcodegen.
Run from the SwiftLMChat/ directory:
    python3 generate_xcodeproj.py
"""

import os, uuid, json, textwrap
from pathlib import Path

def uid():
    """Generate a 24-char uppercase hex UUID matching Xcode format."""
    return uuid.uuid4().hex[:24].upper()

# ── UUIDs ─────────────────────────────────────────────────────────────
PROJ          = uid()
MAIN_GRP      = uid()
SOURCES_GRP   = uid()
VIEWS_GRP     = uid()
VIEWMODELS_GRP= uid()
PRODUCTS_GRP  = uid()
APP_PRODUCT   = uid()
APP_TARGET    = uid()
PHASE_SRC     = uid()
PHASE_RES     = uid()
PHASE_FWK     = uid()
PROJ_CFGLIST  = uid()
PROJ_DEBUG    = uid()
PROJ_RELEASE  = uid()
TGT_CFGLIST   = uid()
TGT_DEBUG     = uid()
TGT_RELEASE   = uid()

# Source files
sources = [
    ("SwiftLMChatApp.swift",              uid(), uid()),  # (name, fileref, buildfile)
    ("Views/RootView.swift",              uid(), uid()),
    ("Views/ChatView.swift",              uid(), uid()),
    ("Views/MessageBubble.swift",         uid(), uid()),
    ("Views/ModelPickerView.swift",       uid(), uid()),
    ("Views/SettingsView.swift",          uid(), uid()),
    ("ViewModels/ChatViewModel.swift",    uid(), uid()),
]
ASSETS_REF  = uid()
ASSETS_BF   = uid()

def pbxproj():
    build_files = ""
    for name, fref, bf in sources:
        build_files += f"\t\t{bf} /* {Path(name).name} in Sources */ = {{isa = PBXBuildFile; fileRef = {fref} /* {Path(name).name} */; }};\n"
    build_files += f"\t\t{ASSETS_BF} /* Assets.xcassets in Resources */ = {{isa = PBXBuildFile; fileRef = {ASSETS_REF} /* Assets.xcassets */; }};\n"

    file_refs = ""
    for name, fref, _ in sources:
        file_refs += f'\t\t{fref} /* {Path(name).name} */ = {{isa = PBXFileReference; lastKnownFileType = sourcecode.swift; path = "{Path(name).name}"; sourceTree = "<group>"; }};\n'
    file_refs += f'\t\t{ASSETS_REF} /* Assets.xcassets */ = {{isa = PBXFileReference; lastKnownFileType = folder.assetcatalog; path = Assets.xcassets; sourceTree = "<group>"; }};\n'
    file_refs += f'\t\t{APP_PRODUCT} /* SwiftLMChat.app */ = {{isa = PBXFileReference; explicitFileType = wrapper.application; includeInIndex = 0; path = SwiftLMChat.app; sourceTree = BUILT_PRODUCTS_DIR; }};\n'

    # Sources group children
    views_children    = "\n".join(f"\t\t\t\t{fref} /* {Path(n).name} */," for n,fref,_ in sources if n.startswith("Views/"))
    viewmodel_children= "\n".join(f"\t\t\t\t{fref} /* {Path(n).name} */," for n,fref,_ in sources if n.startswith("ViewModels/"))
    root_children     = "\n".join(f"\t\t\t\t{fref} /* {Path(n).name} */," for n,fref,_ in sources if "/" not in n)

    # Build phases
    src_children = "\n".join(f"\t\t\t\t{bf} /* {Path(n).name} in Sources */," for n,_,bf in sources)
    
    return f"""// !$*UTF8*$!
{{
\tarchiveVersion = 1;
\tclasses = {{
\t}};
\tobjectVersion = 56;
\tobjects = {{

/* Begin PBXBuildFile section */
{build_files}/* End PBXBuildFile section */

/* Begin PBXFileReference section */
{file_refs}/* End PBXFileReference section */

/* Begin PBXFrameworksBuildPhase section */
\t\t{PHASE_FWK} /* Frameworks */ = {{
\t\t\tisa = PBXFrameworksBuildPhase;
\t\t\tbuildActionMask = 2147483647;
\t\t\tfiles = (
\t\t\t);
\t\t\trunOnlyForDeploymentPostprocessing = 0;
\t\t}};
/* End PBXFrameworksBuildPhase section */

/* Begin PBXGroup section */
\t\t{MAIN_GRP} = {{
\t\t\tisa = PBXGroup;
\t\t\tchildren = (
\t\t\t\t{SOURCES_GRP} /* SwiftLMChat */,
\t\t\t\t{PRODUCTS_GRP} /* Products */,
\t\t\t);
\t\t\tsourceTree = "<group>";
\t\t}};
\t\t{PRODUCTS_GRP} /* Products */ = {{
\t\t\tisa = PBXGroup;
\t\t\tchildren = (
\t\t\t\t{APP_PRODUCT} /* SwiftLMChat.app */,
\t\t\t);
\t\t\tname = Products;
\t\t\tsourceTree = "<group>";
\t\t}};
\t\t{SOURCES_GRP} /* SwiftLMChat */ = {{
\t\t\tisa = PBXGroup;
\t\t\tchildren = (
{root_children}
\t\t\t\t{VIEWS_GRP} /* Views */,
\t\t\t\t{VIEWMODELS_GRP} /* ViewModels */,
\t\t\t\t{ASSETS_REF} /* Assets.xcassets */,
\t\t\t);
\t\t\tpath = SwiftLMChat;
\t\t\tsourceTree = "<group>";
\t\t}};
\t\t{VIEWS_GRP} /* Views */ = {{
\t\t\tisa = PBXGroup;
\t\t\tchildren = (
{views_children}
\t\t\t);
\t\t\tpath = Views;
\t\t\tsourceTree = "<group>";
\t\t}};
\t\t{VIEWMODELS_GRP} /* ViewModels */ = {{
\t\t\tisa = PBXGroup;
\t\t\tchildren = (
{viewmodel_children}
\t\t\t);
\t\t\tpath = ViewModels;
\t\t\tsourceTree = "<group>";
\t\t}};
/* End PBXGroup section */

/* Begin PBXNativeTarget section */
\t\t{APP_TARGET} /* SwiftLMChat */ = {{
\t\t\tisa = PBXNativeTarget;
\t\t\tbuildConfigurationList = {TGT_CFGLIST} /* Build configuration list for PBXNativeTarget "SwiftLMChat" */;
\t\t\tbuildPhases = (
\t\t\t\t{PHASE_SRC} /* Sources */,
\t\t\t\t{PHASE_FWK} /* Frameworks */,
\t\t\t\t{PHASE_RES} /* Resources */,
\t\t\t);
\t\t\tbuildRules = (
\t\t\t);
\t\t\tdependencies = (
\t\t\t);
\t\t\tname = SwiftLMChat;
\t\t\tpackageProductDependencies = (
\t\t\t);
\t\t\tproductName = SwiftLMChat;
\t\t\tproductReference = {APP_PRODUCT} /* SwiftLMChat.app */;
\t\t\tproductType = "com.apple.product-type.application";
\t\t}};
/* End PBXNativeTarget section */

/* Begin PBXProject section */
\t\t{PROJ} /* Project object */ = {{
\t\t\tisa = PBXProject;
\t\t\tattributes = {{
\t\t\t\tBuildIndependentTargetsInParallel = 1;
\t\t\t\tLastSwiftUpdateCheck = 1540;
\t\t\t\tLastUpgradeCheck = 1540;
\t\t\t\tTargetAttributes = {{
\t\t\t\t\t{APP_TARGET} = {{
\t\t\t\t\t\tCreatedOnToolsVersion = 15.4;
\t\t\t\t\t}};
\t\t\t\t}};
\t\t\t}};
\t\t\tbuildConfigurationList = {PROJ_CFGLIST} /* Build configuration list for PBXProject "SwiftLMChat" */;
\t\t\tcompatibilityVersion = "Xcode 14.0";
\t\t\tdevelopmentRegion = en;
\t\t\thasScannedForEncodings = 0;
\t\t\tknownRegions = (
\t\t\t\ten,
\t\t\t\tBase,
\t\t\t);
\t\t\tmainGroup = {MAIN_GRP};
\t\t\tproductsGroup = {PRODUCTS_GRP} /* Products */;
\t\t\tprojectDirPath = "";
\t\t\tprojectRoot = "";
\t\t\ttargets = (
\t\t\t\t{APP_TARGET} /* SwiftLMChat */,
\t\t\t);
\t\t}};
/* End PBXProject section */

/* Begin PBXResourcesBuildPhase section */
\t\t{PHASE_RES} /* Resources */ = {{
\t\t\tisa = PBXResourcesBuildPhase;
\t\t\tbuildActionMask = 2147483647;
\t\t\tfiles = (
\t\t\t\t{ASSETS_BF} /* Assets.xcassets in Resources */,
\t\t\t);
\t\t\trunOnlyForDeploymentPostprocessing = 0;
\t\t}};
/* End PBXResourcesBuildPhase section */

/* Begin PBXSourcesBuildPhase section */
\t\t{PHASE_SRC} /* Sources */ = {{
\t\t\tisa = PBXSourcesBuildPhase;
\t\t\tbuildActionMask = 2147483647;
\t\t\tfiles = (
{src_children}
\t\t\t);
\t\t\trunOnlyForDeploymentPostprocessing = 0;
\t\t}};
/* End PBXSourcesBuildPhase section */

/* Begin XCBuildConfiguration section */
\t\t{PROJ_DEBUG} /* Debug */ = {{
\t\t\tisa = XCBuildConfiguration;
\t\t\tbuildSettings = {{
\t\t\t\tALWAYS_SEARCH_USER_PATHS = NO;
\t\t\t\tCLANG_ENABLE_MODULES = YES;
\t\t\t\tCOPY_PHASE_STRIP = NO;
\t\t\t\tDEBUG_INFORMATION_FORMAT = dwarf;
\t\t\t\tENABLE_TESTABILITY = YES;
\t\t\t\tGCC_DYNAMIC_NO_PIC = NO;
\t\t\t\tGCC_OPTIMIZATION_LEVEL = 0;
\t\t\t\tMTL_ENABLE_DEBUG_INFO = INCLUDE_SOURCE;
\t\t\t\tMTL_FAST_MATH = YES;
\t\t\t\tONLY_ACTIVE_ARCH = YES;
\t\t\t\tSWIFT_ACTIVE_COMPILATION_CONDITIONS = DEBUG;
\t\t\t}};
\t\t\tname = Debug;
\t\t}};
\t\t{PROJ_RELEASE} /* Release */ = {{
\t\t\tisa = XCBuildConfiguration;
\t\t\tbuildSettings = {{
\t\t\t\tALWAYS_SEARCH_USER_PATHS = NO;
\t\t\t\tCLANG_ENABLE_MODULES = YES;
\t\t\t\tCOPY_PHASE_STRIP = NO;
\t\t\t\tDEBUG_INFORMATION_FORMAT = "dwarf-with-dsym";
\t\t\t\tMTL_FAST_MATH = YES;
\t\t\t\tSWIFT_COMPILATION_MODE = wholemodule;
\t\t\t}};
\t\t\tname = Release;
\t\t}};
\t\t{TGT_DEBUG} /* Debug */ = {{
\t\t\tisa = XCBuildConfiguration;
\t\t\tbuildSettings = {{
\t\t\t\tASSTCATALOG_COMPILER_APPICON_NAME = AppIcon;
\t\t\t\tASSTCATALOG_COMPILER_GLOBAL_ACCENT_COLOR_NAME = AccentColor;
\t\t\t\tCODE_SIGN_STYLE = Automatic;
\t\t\t\tCURRENT_PROJECT_VERSION = 1;
\t\t\t\tGENERATE_INFOPLIST_FILE = YES;
\t\t\t\tINFOPLIST_KEY_CFBundleDisplayName = "SwiftLM Chat";
\t\t\t\tINFOPLIST_KEY_NSHumanReadableCopyright = "";
\t\t\t\tINFOPLIST_KEY_UIApplicationSceneManifest_Generation = YES;
\t\t\t\tINFOPLIST_KEY_UIApplicationSupportsIndirectInputEvents = YES;
\t\t\t\tINFOPLIST_KEY_UILaunchScreen_Generation = YES;
\t\t\t\tINFOPLIST_KEY_UISupportedInterfaceOrientations_iPad = "UIInterfaceOrientationPortrait UIInterfaceOrientationPortraitUpsideDown UIInterfaceOrientationLandscapeLeft UIInterfaceOrientationLandscapeRight";
\t\t\t\tINFOPLIST_KEY_UISupportedInterfaceOrientations_iPhone = "UIInterfaceOrientationPortrait UIInterfaceOrientationLandscapeLeft UIInterfaceOrientationLandscapeRight";
\t\t\t\tIPHONEOS_DEPLOYMENT_TARGET = 17.0;
\t\t\t\tLE_SWIFT_VERSION = 5.9;
\t\t\t\tMARKETING_VERSION = 1.0;
\t\t\t\tPRODUCT_BUNDLE_IDENTIFIER = com.sharpai.SwiftLMChat;
\t\t\t\tPRODUCT_NAME = "$(TARGET_NAME)";
\t\t\t\tSDKROOT = auto;
\t\t\t\tSUPPORTED_PLATFORMS = "iphoneos iphonesimulator macosx";
\t\t\t\tSWIFT_EMIT_LOC_STRINGS = YES;
\t\t\t\tSWIFT_VERSION = 5.9;
\t\t\t\tTARGETED_DEVICE_FAMILY = "1,2";
\t\t\t}};
\t\t\tname = Debug;
\t\t}};
\t\t{TGT_RELEASE} /* Release */ = {{
\t\t\tisa = XCBuildConfiguration;
\t\t\tbuildSettings = {{
\t\t\t\tASSTCATALOG_COMPILER_APPICON_NAME = AppIcon;
\t\t\t\tASSTCATALOG_COMPILER_GLOBAL_ACCENT_COLOR_NAME = AccentColor;
\t\t\t\tCODE_SIGN_STYLE = Automatic;
\t\t\t\tCURRENT_PROJECT_VERSION = 1;
\t\t\t\tGENERATE_INFOPLIST_FILE = YES;
\t\t\t\tINFOPLIST_KEY_CFBundleDisplayName = "SwiftLM Chat";
\t\t\t\tIPHONEOS_DEPLOYMENT_TARGET = 17.0;
\t\t\t\tLE_SWIFT_VERSION = 5.9;
\t\t\t\tMARKETING_VERSION = 1.0;
\t\t\t\tPRODUCT_BUNDLE_IDENTIFIER = com.sharpai.SwiftLMChat;
\t\t\t\tPRODUCT_NAME = "$(TARGET_NAME)";
\t\t\t\tSDKROOT = auto;
\t\t\t\tSUPPORTED_PLATFORMS = "iphoneos iphonesimulator macosx";
\t\t\t\tSWIFT_EMIT_LOC_STRINGS = YES;
\t\t\t\tSWIFT_VERSION = 5.9;
\t\t\t\tTARGETED_DEVICE_FAMILY = "1,2";
\t\t\t}};
\t\t\tname = Release;
\t\t}};
/* End XCBuildConfiguration section */

/* Begin XCConfigurationList section */
\t\t{PROJ_CFGLIST} /* Build configuration list for PBXProject "SwiftLMChat" */ = {{
\t\t\tisa = XCConfigurationList;
\t\t\tbuildConfigurations = (
\t\t\t\t{PROJ_DEBUG} /* Debug */,
\t\t\t\t{PROJ_RELEASE} /* Release */,
\t\t\t);
\t\t\tdefaultConfigurationIsVisible = 0;
\t\t\tdefaultConfigurationName = Release;
\t\t}};
\t\t{TGT_CFGLIST} /* Build configuration list for PBXNativeTarget "SwiftLMChat" */ = {{
\t\t\tisa = XCConfigurationList;
\t\t\tbuildConfigurations = (
\t\t\t\t{TGT_DEBUG} /* Debug */,
\t\t\t\t{TGT_RELEASE} /* Release */,
\t\t\t);
\t\t\tdefaultConfigurationIsVisible = 0;
\t\t\tdefaultConfigurationName = Release;
\t\t}};
/* End XCConfigurationList section */
\t}};
\trootObject = {PROJ} /* Project object */;
}}
"""

def main():
    proj_dir = Path("SwiftLMChat.xcodeproj")
    proj_dir.mkdir(exist_ok=True)

    pbx_path = proj_dir / "project.pbxproj"
    pbx_path.write_text(pbxproj())
    print(f"✅  Generated {pbx_path}")

    # SPM package references (local paths)
    pkg_refs = {
      "object": {
        "pins": [
          {"identity": "mlx-swift",     "kind": "localSourceControl", "location": "../LocalPackages/mlx-swift"},
          {"identity": "mlx-swift-lm",  "kind": "localSourceControl", "location": "../mlx-swift-lm"},
        ],
        "version": 1
      }
    }
    ws_dir = proj_dir / "project.xcworkspace"
    ws_dir.mkdir(exist_ok=True)
    (ws_dir / "contents.xcworkspacedata").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Workspace version = "1.0">\n'
        '   <FileRef location = "self:"></FileRef>\n'
        '</Workspace>\n'
    )
    print(f"✅  Generated workspace data")
    print("\n🎉 Done! Open SwiftLMChat.xcodeproj in Xcode.")
    print("   Then add SPM dependencies via File → Add Package Dependencies:")
    print("   • ../LocalPackages/mlx-swift")
    print("   • ../mlx-swift-lm")

if __name__ == "__main__":
    main()

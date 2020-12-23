import React from 'react';
import Logo from './Logo/Logo';
import './Navigation.css';

const Navigation = () => {
    return (
<div class="navbar">
 <ul className='List'>
 <Logo />
 <li className='w-20 f3 link dim white underline pa3 pointer'>Sign Out</li>
 </ul>
 
</div>
    );
}

export default Navigation;